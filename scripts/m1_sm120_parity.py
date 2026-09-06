import argparse
import sys
import types

import torch

from sglang.kernels.ops.speculative.dspark.dspark_accept import (
    AcceptGreedy,
    AcceptSampling,
    CapCorrectLen,
    FinalizeAcceptLens,
    SelectMixedAccept,
    SoftmaxTemp,
)
from sglang.kernels.ops.speculative.dspark.dspark_attn_metadata import (
    build_dspark_swa_page_indices,
    build_dspark_swa_page_indices_triton,
    compute_dspark_window_gather,
)
from sglang.kernels.ops.speculative.dspark.dspark_draft_model import (
    BuildStepLocal,
    SampleStepTokens,
)
from sglang.kernels.ops.speculative.dspark.dspark_schedule import (
    ScheduleVerifyLensTopk,
)
from sglang.kernels.ops.speculative.dspark.dspark_verify_window import (
    BuildOutTokens,
    CompactRowIndex,
)
from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2
from sglang.srt.speculative.dspark_components.dspark_planner import (
    DSparkScheduleConfig,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    parser.add_argument("--device", default="cuda:0", help="CUDA device to run on")
    parser.add_argument("--bs", type=int, default=8, help="batch size for the sweeps")
    parser.add_argument("--vocab", type=int, default=2048, help="vocabulary width (kept small for sweep speed)")
    parser.add_argument("--gamma", type=int, default=5, help="DSpark block size (checkpoint native: 5)")
    parser.add_argument("--seed", type=int, default=0, help="base seed; each section derives its own")
    parser.add_argument("--tol", type=float, default=1e-5, help="relative tolerance for float comparisons")
    return parser.parse_args()


class ParityReport:
    def __init__(self, tol: float) -> None:
        self.tol = tol
        self.failures = []

    def compare(self, name: str, outputs: dict, reference: dict) -> None:
        for key in reference:
            got, want = outputs[key], reference[key]
            if got.shape != want.shape:
                print(f"[parity] FAIL {name}.{key}: shape {tuple(got.shape)} != reference {tuple(want.shape)}")
                self.failures.append(f"{name}.{key}")
                continue
            if not torch.is_floating_point(got):
                exact = torch.equal(got, want)
                if exact:
                    print(f"[parity] ok {name}.{key}: exact ({tuple(got.shape)}, {got.dtype})")
                else:
                    diff = (got != want).sum().item()
                    print(f"[parity] FAIL {name}.{key}: {diff}/{got.numel()} mismatching elements ({tuple(got.shape)}, {got.dtype})")
                    self.failures.append(f"{name}.{key}")
                continue
            dev = (got.float() - want.float()).abs().max().item()
            ok = torch.allclose(got, want, rtol=self.tol, atol=1e-6)
            if ok:
                print(f"[parity] ok {name}.{key}: allclose, max|diff|={dev:.3e} ({tuple(got.shape)}, {got.dtype})")
            else:
                scale = want.float().abs().max().item() or 1.0
                print(f"[parity] FAIL {name}.{key}: max|diff|={dev:.3e} scale={scale:.3e} ({tuple(got.shape)}, {got.dtype})")
                self.failures.append(f"{name}.{key}")


def sweep_softmax_temp(report: ParityReport, device: str, bs: int, vocab: int, rows: int, seed: int) -> None:
    g = torch.Generator(device="cpu").manual_seed(seed)
    logits = torch.randn(bs * rows, vocab, generator=g).to(device)
    temperatures = torch.rand(bs, generator=g).mul(0.99).add(0.01).to(device)
    ref = SoftmaxTemp.torch(logits=logits, temperatures=temperatures, rows_per_request=rows)
    out = SoftmaxTemp.triton(logits=logits, temperatures=temperatures, rows_per_request=rows)
    print(f"[parity] softmax_temp: logits={tuple(logits.shape)} temperatures={tuple(temperatures.shape)} rows_per_request={rows}")
    report.compare("softmax_temp", {"probs": out}, {"probs": ref})


def sweep_sample_step_tokens(report: ParityReport, device: str, n: int, vocab: int, seed: int) -> None:
    g = torch.Generator(device="cpu").manual_seed(seed)
    step_logits = torch.randn(n, vocab, generator=g).to(device)
    temperatures = torch.rand(n, generator=g).mul(0.99).add(0.01).to(device)
    greedy_mask = (torch.rand(n, generator=g) < 0.3).to(device)
    exp_noise = torch.rand(n, vocab, generator=g).mul(0.99).add(0.01).log().to(device)
    ref = SampleStepTokens.torch(
        step_logits=step_logits, temperatures=temperatures, greedy_mask=greedy_mask, exp_noise=exp_noise
    )
    out = SampleStepTokens.triton(
        step_logits=step_logits, temperatures=temperatures, greedy_mask=greedy_mask, exp_noise=exp_noise
    )
    print(f"[parity] sample_step_tokens: rows={n} vocab={vocab} greedy_rows={int(greedy_mask.sum())}")
    report.compare("sample_step_tokens", {"tokens": out}, {"tokens": ref})


def sweep_build_step_local(report: ParityReport, device: str, n: int, width: int, pad_to: int, seed: int) -> None:
    g = torch.Generator(device="cpu").manual_seed(seed)
    bias = torch.randn(n, width, generator=g).to(device)
    base_local = torch.randn(n, pad_to, generator=g).to(device)
    ref = BuildStepLocal.torch(bias=bias, base_local=base_local)
    out = BuildStepLocal.triton(bias=bias, base_local=base_local)
    print(f"[parity] build_step_local: bias={tuple(bias.shape)} base_local={tuple(base_local.shape)}")
    report.compare("build_step_local", {"local": out}, {"local": ref})


def _verify_inputs(g: torch.Generator, bs: int, slots: int, vocab: int, device: str) -> dict:
    target_logits = torch.randn(bs * slots, vocab, generator=g)
    target_predict = target_logits.view(bs, slots, vocab).argmax(dim=-1)
    candidates = torch.randint(0, vocab, (bs, slots), generator=g)
    match_rows = int(bs * 0.5)
    for t in range(slots - 1):
        keep = torch.rand(bs, generator=g) < (0.8 if t < 3 else 0.3)
        apply_rows = torch.arange(match_rows)[keep[:match_rows]]
        candidates[apply_rows, t + 1] = target_predict[apply_rows, t]
    draft_probs = torch.softmax(torch.randn(bs * slots, vocab, generator=g), dim=-1).view(bs, slots, vocab)
    return {
        "candidates": candidates.to(device),
        "target_logits": target_logits.to(device),
        "draft_probs": draft_probs.to(device),
    }


def sweep_accept_greedy(report: ParityReport, device: str, bs: int, slots: int, vocab: int, seed: int) -> None:
    g = torch.Generator(device="cpu").manual_seed(seed)
    inputs = _verify_inputs(g, bs, slots, vocab, device)
    for label, cutoff in (("uncapped", None), ("capped", torch.randint(1, slots + 1, (bs,), generator=g).to(device))):
        ref = AcceptGreedy.torch(
            candidates=inputs["candidates"],
            target_logits=inputs["target_logits"],
            verify_num_draft_tokens=slots,
            cutoff_verify_lens=cutoff,
        )
        out = AcceptGreedy.triton(
            candidates=inputs["candidates"],
            target_logits=inputs["target_logits"],
            verify_num_draft_tokens=slots,
            cutoff_verify_lens=cutoff,
        )
        print(f"[parity] accept_greedy/{label}: candidates={tuple(inputs['candidates'].shape)} cutoff={'none' if cutoff is None else tuple(cutoff.shape)}")
        report.compare(
            f"accept_greedy_{label}",
            {"correct_len": out[0], "bonus": out[1], "cap_trim_lens": out[2]},
            {"correct_len": ref[0], "bonus": ref[1], "cap_trim_lens": ref[2]},
        )


def sweep_accept_sampling(report: ParityReport, device: str, bs: int, gamma: int, slots: int, vocab: int, seed: int) -> None:
    g = torch.Generator(device="cpu").manual_seed(seed)
    inputs = _verify_inputs(g, bs, slots, vocab, device)
    sampling_info = types.SimpleNamespace(
        need_top_k_sampling=False,
        need_top_p_sampling=False,
        temperatures=torch.rand(bs, 1, generator=g).mul(0.99).add(0.01).to(device),
    )
    # Legacy Eagle-shaped fields, documented upstream as unused for DFLASH
    # (draft state relays via FutureMap); the accept kernels read only
    # max_top_k and uniform_top_k_value, so zero-filled stand-ins satisfy the
    # dataclass constructor without participating in the parity math.
    draft_input = DFlashDraftInputV2(
        topk_p=torch.zeros(bs, 1, device=device),
        topk_index=torch.zeros(bs, 1, dtype=torch.int64, device=device),
        bonus_tokens=torch.zeros(bs, dtype=torch.int64, device=device),
        new_seq_lens=torch.zeros(bs, dtype=torch.int64, device=device),
        hidden_states=torch.zeros(bs, 1, device=device),
        max_top_k=1,
        uniform_top_k_value=None,
    )
    results = {}
    for path in ("ref", "out"):
        torch.cuda.manual_seed_all(seed)
        results[path] = (AcceptSampling.torch if path == "ref" else AcceptSampling.triton)(
            candidates=inputs["candidates"],
            target_logits=inputs["target_logits"],
            draft_probs=inputs["draft_probs"],
            sampling_info=sampling_info,
            draft_input=draft_input,
            gamma=gamma,
            verify_num_draft_tokens=slots,
            cutoff_verify_lens=None,
        )
    print(f"[parity] accept_sampling: candidates={tuple(inputs['candidates'].shape)} draft_probs={tuple(inputs['draft_probs'].shape)}")
    report.compare(
        "accept_sampling",
        {"correct_len": results["out"][0], "bonus": results["out"][1], "cap_trim_lens": results["out"][2]},
        {"correct_len": results["ref"][0], "bonus": results["ref"][1], "cap_trim_lens": results["ref"][2]},
    )


def sweep_accept_bookkeeping(report: ParityReport, device: str, bs: int, slots: int, gamma: int, vocab: int, seed: int) -> None:
    g = torch.Generator(device="cpu").manual_seed(seed)
    mixed = dict(
        greedy_mask=(torch.rand(bs, generator=g) < 0.4).to(device),
        greedy_len=torch.randint(0, slots, (bs,), generator=g).to(device),
        greedy_bonus=torch.randint(0, vocab, (bs,), generator=g).to(device),
        greedy_trim=torch.randint(0, slots, (bs,), generator=g).to(device),
        sampling_len=torch.randint(0, slots, (bs,), generator=g).to(device),
        sampling_bonus=torch.randint(0, vocab, (bs,), generator=g).to(device),
        sampling_trim=torch.randint(0, slots, (bs,), generator=g).to(device),
    )
    print(f"[parity] select_mixed_accept: bs={bs} greedy_rows={int(mixed['greedy_mask'].sum())}")
    ref_sel = SelectMixedAccept.torch(**mixed)
    out_sel = SelectMixedAccept.triton(**mixed)
    report.compare(
        "select_mixed_accept",
        {"correct_len": out_sel.correct_len, "bonus": out_sel.bonus, "cap_trim_lens": out_sel.cap_trim_lens},
        {"correct_len": ref_sel.correct_len, "bonus": ref_sel.bonus, "cap_trim_lens": ref_sel.cap_trim_lens},
    )

    correct_len = torch.randint(0, slots, (bs,), generator=g).to(device)
    verify_lens = torch.randint(1, slots + 1, (bs,), generator=g).to(device)
    ref_cap = CapCorrectLen.torch(correct_len=correct_len, verify_lens=verify_lens)
    out_cap = CapCorrectLen.triton(correct_len=correct_len, verify_lens=verify_lens)
    report.compare(
        "cap_correct_len",
        {"capped": out_cap[0], "trim": out_cap[1]},
        {"capped": ref_cap[0], "trim": ref_cap[1]},
    )

    prefix_lens = torch.randint(10, 1000, (bs,), generator=g).to(device)
    ref_fin = FinalizeAcceptLens.torch(correct_len=correct_len, cap_trim_lens=ref_cap[1], prefix_lens=prefix_lens)
    out_fin = FinalizeAcceptLens.triton(correct_len=correct_len, cap_trim_lens=ref_cap[1], prefix_lens=prefix_lens)
    report.compare(
        "finalize_accept_lens",
        {"commit_lens": out_fin.commit_lens, "new_seq_lens": out_fin.new_seq_lens, "cap_trim_lens": out_fin.cap_trim_lens},
        {"commit_lens": ref_fin.commit_lens, "new_seq_lens": ref_fin.new_seq_lens, "cap_trim_lens": ref_fin.cap_trim_lens},
    )

    draft_tokens = torch.randint(0, vocab, (bs, gamma), generator=g).to(device)
    bonus = torch.randint(0, vocab, (bs,), generator=g).to(device)
    ref_tokens = BuildOutTokens.torch(
        draft_tokens=draft_tokens, correct_len=correct_len, bonus=bonus, verify_num_draft_tokens=slots, gamma=gamma
    )
    out_tokens = BuildOutTokens.triton(
        draft_tokens=draft_tokens, correct_len=correct_len, bonus=bonus, verify_num_draft_tokens=slots, gamma=gamma
    )
    report.compare("build_out_tokens", {"tokens": out_tokens}, {"tokens": ref_tokens})


def sweep_compact_row_index(report: ParityReport, device: str, bs: int, slots: int, seed: int) -> None:
    g = torch.Generator(device="cpu").manual_seed(seed)
    verify_lens = torch.randint(1, slots + 1, (bs,), generator=g).to(device)
    padded_total = int(verify_lens.sum().item()) + 17
    ref = CompactRowIndex.torch(verify_lens=verify_lens, padded_total=padded_total, device=device)
    out = CompactRowIndex.triton(verify_lens=verify_lens, padded_total=padded_total, device=device)
    print(f"[parity] compact_row_index: verify_lens={tuple(verify_lens.shape)} padded_total={padded_total}")
    report.compare(
        "compact_row_index",
        {"req_id": out[0], "within": out[1], "valid": out[2]},
        {"req_id": ref[0], "within": ref[1], "valid": ref[2]},
    )


def sweep_schedule(report: ParityReport, device: str, bs: int, gamma: int, seed: int) -> None:
    g = torch.Generator(device="cpu").manual_seed(seed)
    confidence = torch.rand(bs, gamma, generator=g).mul(0.98).add(0.01).to(device)
    cfg = DSparkScheduleConfig(gamma=gamma)
    cfg.validate()
    budget = max(bs, (bs * gamma) // 2)
    ref = ScheduleVerifyLensTopk.torch(confidence=confidence, budget=budget, cfg=cfg)
    out = ScheduleVerifyLensTopk.triton(confidence=confidence, budget=budget, cfg=cfg)
    print(f"[parity] schedule_verify_lens_topk: confidence={tuple(confidence.shape)} budget={budget} max_len={cfg.resolved_max_verify_len()}")
    report.compare("schedule_verify_lens_topk", {"verify_lens": out}, {"verify_lens": ref})


def sweep_swa_page_indices(report: ParityReport, device: str, bs: int, gamma: int, seed: int) -> None:
    g = torch.Generator(device="cpu").manual_seed(seed)
    slots = gamma + 1
    swa_window = 128
    pool_size = 64
    max_len = 4096
    n_tokens = 1 << 16
    req_to_token = torch.randint(1, n_tokens, (pool_size, max_len), generator=g).to(torch.int64).to(device)
    full_to_swa = torch.randint(0, n_tokens, (n_tokens + 1,), generator=g).to(torch.int32).to(device)
    seq_lens = torch.randint(swa_window + 2, 512, (bs * slots,), generator=g).to(torch.int64)
    seq_lens_casual = seq_lens.to(device)
    req_pool = torch.arange(bs).remainder(pool_size).repeat_interleave(slots).to(device)
    gather = compute_dspark_window_gather(
        seq_lens_casual=seq_lens_casual,
        req_pool_indices_repeated=req_pool,
        block_size=slots,
        swa_window=swa_window,
    )
    out_loc = torch.randint(1, n_tokens, (bs * slots,), generator=g).to(torch.int64).to(device)
    common = dict(
        req_to_token=req_to_token,
        full_to_swa_mapping=full_to_swa,
        req_pool_indices_per_request=gather.req_pool_indices_per_request,
        offsets=gather.offsets,
        out_loc=out_loc,
        context_lens=gather.context_lens,
        block_size=slots,
        swa_window=swa_window,
        page_index_aligned_size=64,
    )
    ref = build_dspark_swa_page_indices(invalid=gather.invalid, **common)
    out = build_dspark_swa_page_indices_triton(**common)
    width = ref[0].shape[-1]
    print(f"[parity] swa_page_indices: bs={bs} slots={slots} swa_window={swa_window} target_width={width}")
    report.compare(
        "swa_page_indices",
        {"indices": out[0], "topk_lengths": out[1]},
        {"indices": ref[0], "topk_lengths": ref[1]},
    )
    pad = ref[0][:, -1]
    if width > swa_window and (pad != -1).any():
        print(f"[parity] FAIL swa_page_indices.padding: {int((pad != -1).sum())} rows have non -1 padding in the tail column")
        report.failures.append("swa_page_indices.padding")


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        print("[parity] FATAL: CUDA is required (the Triton paths are the objects under test)")
        sys.exit(1)
    device = args.device
    torch.cuda.set_device(device)
    props = torch.cuda.get_device_properties(device)
    print(f"[parity] device {device}: {props.name} cc={props.major}.{props.minor}")
    print(f"[parity] torch={torch.__version__} params: bs={args.bs} vocab={args.vocab} gamma={args.gamma} seed={args.seed} tol={args.tol}")

    report = ParityReport(tol=args.tol)
    slots = args.gamma + 1
    sweeps = [
        ("softmax_temp", lambda: sweep_softmax_temp(report, device, args.bs, args.vocab, slots, args.seed + 1)),
        ("sample_step_tokens", lambda: sweep_sample_step_tokens(report, device, args.bs * slots, args.vocab, args.seed + 2)),
        ("build_step_local", lambda: sweep_build_step_local(report, device, args.bs * slots, 96, 128, args.seed + 3)),
        ("accept_greedy", lambda: sweep_accept_greedy(report, device, args.bs, slots, args.vocab, args.seed + 4)),
        ("accept_sampling", lambda: sweep_accept_sampling(report, device, args.bs, args.gamma, slots, args.vocab, args.seed + 5)),
        ("accept_bookkeeping", lambda: sweep_accept_bookkeeping(report, device, args.bs, slots, args.gamma, args.vocab, args.seed + 6)),
        ("compact_row_index", lambda: sweep_compact_row_index(report, device, args.bs, slots, args.seed + 7)),
        ("schedule_verify_lens_topk", lambda: sweep_schedule(report, device, args.bs, args.gamma, args.seed + 8)),
        ("swa_page_indices", lambda: sweep_swa_page_indices(report, device, args.bs, args.gamma, args.seed + 9)),
    ]
    for name, run in sweeps:
        try:
            run()
        except Exception:
            print(f"[parity] FAIL {name}: raised during execution")
            raise

    if report.failures:
        print(f"[parity] FAILED: {len(report.failures)} output(s) diverged: {report.failures}")
        sys.exit(1)
    print("[parity] all kernels match their PyTorch references on this device")


if __name__ == "__main__":
    main()
