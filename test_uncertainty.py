"""
Test script for the uncertainty-based recursive reasoning method.

Runs several problems with sampling-based uncertainty estimation and
evaluates whether the confidence scores are meaningful.

Usage:
  python test_uncertainty.py [provider] [num_samples]
  python test_uncertainty.py mistral 3
"""

import re
import sys
import time
import json
from dotenv import load_dotenv

load_dotenv()

from model import create_model
from state import ThoughtState
from controller import Controller, ControllerConfig
from logger import ReasoningLogger
from uncertainty import UncertaintyEstimator, UncertaintyResult
from prompt import build_prompt
from parser import parse_model_output


def extract_boxed_answer(solution: str) -> str:
    match = re.search(r"\\boxed\{([^}]+)\}", solution)
    if match:
        return match.group(1).strip()
    nums = re.findall(r"[\d,.]+", solution)
    if nums:
        return nums[-1].strip()
    return solution.strip()


def normalize_answer(answer: str) -> str:
    answer = answer.strip().lower()
    answer = answer.replace(",", "")
    answer = answer.replace(" ", "")
    try:
        return str(float(answer))
    except ValueError:
        return answer


def smart_comparator(a: str, b: str) -> bool:
    return normalize_answer(a) == normalize_answer(b)


TEST_PROBLEMS = [
    {
        "name": "Easy arithmetic",
        "problem": "What is 7 * 8?",
        "expected_answer": "56",
        "expect_high": True,
    },
    {
        "name": "Moderate algebra",
        "problem": "Solve for x: 3x^2 - 12x + 9 = 0. Give both roots.",
        "expected_answer": "1, 3",
        "expect_high": True,
    },
    {
        "name": "Tricky combinatorics",
        "problem": "How many ways can you arrange the letters in the word MISSISSIPPI?",
        "expected_answer": "34650",
        "expect_high": True,
    },
]


def print_banner(text: str):
    print()
    print("=" * 70)
    print(f"  {text}")
    print("=" * 70)


def compute_agreement_offline(answers, extractor=None, comparator=None):
    """Re-compute agreement on a set of answers with different extractor/comparator."""
    if extractor is None:
        extractor = lambda x: x
    if comparator is None:
        comparator = lambda a, b: a == b

    extracted = []
    for a in answers:
        if a is not None:
            extracted.append(extractor(a))
        else:
            extracted.append(None)

    valid = [e for e in extracted if e is not None]
    if not valid:
        return 0.0, 0.0, {}, None

    clusters = []
    for ans in valid:
        matched = False
        for cluster in clusters:
            if comparator(ans, cluster[0]):
                cluster[1].append(ans)
                matched = True
                break
        if not matched:
            clusters.append((ans, [ans]))

    clusters.sort(key=lambda c: len(c[1]), reverse=True)
    majority = clusters[0][0]
    majority_count = len(clusters[0][1])
    agreement_ratio = majority_count / len(valid)
    measured_confidence = majority_count / len(extracted)

    counts = {}
    for canonical, members in clusters:
        counts[str(canonical)] = len(members)
    failed = sum(1 for e in extracted if e is None)
    if failed > 0:
        counts["_failed"] = failed

    return measured_confidence, agreement_ratio, counts, majority


def sample_with_rate_limit(model, system_prompt, user_prompt, num_samples, delay=2.0):
    """Generate N samples with a delay between calls to avoid rate limits."""
    samples = []
    answers = []
    self_confs = []

    for i in range(num_samples):
        if i > 0:
            time.sleep(delay)
        try:
            response = model.generate(system_prompt, user_prompt, json_mode=True)
            parse_result = parse_model_output(response.text)
            samples.append(parse_result)

            if parse_result.success:
                solution = parse_result.updated_state.get("current_solution", "")
                conf = float(parse_result.updated_state.get("confidence", 0.0))
                answers.append(solution)
                self_confs.append(conf)
            else:
                answers.append(None)
                self_confs.append(0.0)
        except Exception as e:
            print(f"    ⚠ Sample {i+1} failed: {e}")
            answers.append(None)
            self_confs.append(0.0)
            samples.append(None)

    return samples, answers, self_confs


def main():
    provider = "mistral"
    num_samples = 5
    delay = 1.0

    if len(sys.argv) > 1:
        provider = sys.argv[1]
    if len(sys.argv) > 2:
        num_samples = int(sys.argv[2])

    if provider == "gemini":
        delay = 15.0
        print("NOTE: Using Gemini free tier — adding 15s delay between samples")

    print_banner(f"Uncertainty Test Suite  (provider={provider}, N={num_samples})")

    model = create_model(provider=provider)

    results_summary = []

    for info in TEST_PROBLEMS:
        print(f"\n{'─' * 70}")
        print(f"  {info['name']}: {info['problem']}")
        print(f"  Expected: {info['expected_answer']}")
        print(f"{'─' * 70}")

        state = ThoughtState(problem=info["problem"])
        system_prompt, user_prompt = build_prompt(state)

        t0 = time.time()
        samples, answers, self_confs = sample_with_rate_limit(
            model, system_prompt, user_prompt, num_samples, delay=delay
        )
        elapsed = time.time() - t0

        # ── Raw string match (current default behavior) ──────────────────
        raw_conf, raw_agree, raw_counts, raw_majority = compute_agreement_offline(
            answers, extractor=None, comparator=None
        )

        # ── Smart: extract \boxed{} and normalize ────────────────────────
        smart_conf, smart_agree, smart_counts, smart_majority = compute_agreement_offline(
            answers, extractor=extract_boxed_answer, comparator=smart_comparator
        )

        print(f"\n  Samples collected: {len(answers)} ({sum(1 for a in answers if a is not None)} valid)")
        print(f"  Self-reported confidences: {self_confs}")
        avg_self = sum(self_confs) / len(self_confs) if self_confs else 0
        print(f"  Avg self-reported: {avg_self:.2f}")

        print(f"\n  [DEFAULT — raw string match]")
        print(f"    measured_confidence : {raw_conf:.2f}")
        print(f"    agreement_ratio    : {raw_agree:.2f}")
        print(f"    clusters           : {raw_counts}")

        print(f"\n  [SMART — \\boxed{{}} extractor + normalized compare]")
        print(f"    measured_confidence : {smart_conf:.2f}")
        print(f"    agreement_ratio    : {smart_agree:.2f}")
        print(f"    majority_answer    : {smart_majority}")
        print(f"    clusters           : {smart_counts}")

        correct_answer = info["expected_answer"]
        if smart_majority is not None:
            ans_correct = normalize_answer(str(smart_majority)) == normalize_answer(correct_answer) or \
                          correct_answer.lower() in str(smart_majority).lower()
        else:
            ans_correct = False

        if info["expect_high"]:
            if smart_conf >= 0.6 and ans_correct:
                verdict = "✅ PASS — high confidence, correct answer"
            elif smart_conf >= 0.6 and not ans_correct:
                verdict = "⚠️  High confidence but WRONG answer"
            elif smart_conf < 0.6 and ans_correct:
                verdict = "⚠️  Correct answer but LOW confidence (extractor may need tuning)"
            else:
                verdict = "❌ FAIL — low confidence AND wrong answer"
        else:
            verdict = f"ℹ️  conf={smart_conf:.2f}, correct={ans_correct}"

        raw_vs_smart = ""
        if raw_conf < 0.5 and smart_conf >= 0.8:
            raw_vs_smart = " 🔴 RAW underestimates confidence (answer_extractor needed!)"
        elif abs(raw_conf - smart_conf) < 0.1:
            raw_vs_smart = " ✅ RAW ≈ SMART (phrasing was consistent)"

        print(f"\n  Verdict     : {verdict}")
        print(f"  Raw vs Smart: {raw_vs_smart}")
        print(f"  Time        : {elapsed:.1f}s")

        results_summary.append({
            "problem": info["name"],
            "expected": info["expected_answer"],
            "raw_conf": raw_conf,
            "smart_conf": smart_conf,
            "majority": smart_majority,
            "correct": ans_correct,
            "self_reported_avg": avg_self,
            "verdict": verdict,
        })

    # ── Part 2: Full controller run ──────────────────────────────────────
    print_banner("Full Controller Loop (smart extractor, easy problem)")
    easy = TEST_PROBLEMS[0]

    if provider == "gemini":
        print("  Waiting 60s for Gemini rate limit cooldown...")
        time.sleep(60)

    log_file = "test_uncertainty_controller.jsonl"
    config = ControllerConfig(
        max_steps=3,
        confidence_threshold=0.9,
        use_uncertainty=True,
        uncertainty_samples=num_samples,
        answer_extractor=extract_boxed_answer,
    )

    logger = ReasoningLogger(log_path=log_file)
    controller = Controller(model=model, config=config, logger=logger)
    controller.uncertainty_estimator.answer_comparator = smart_comparator

    t0 = time.time()
    final_state = controller.run(easy["problem"])
    elapsed = time.time() - t0

    print(f"\n  Final solution   : {final_state.current_solution[:150]}")
    print(f"  Final confidence : {final_state.confidence:.2f}")
    print(f"  Total steps      : {final_state.step}")
    print(f"  Time             : {elapsed:.1f}s")

    print(f"\n  Logged uncertainty per step:")
    with open(log_file) as f:
        for line in f:
            entry = json.loads(line)
            if "uncertainty_detail" in entry and entry["uncertainty_detail"]:
                ud = entry["uncertainty_detail"]
                step = entry.get("step", "?")
                print(f"    Step {step}: measured={ud['measured_confidence']:.2f}, "
                      f"agreement={ud['agreement_ratio']:.2f}, "
                      f"clusters={ud['answer_counts']}, "
                      f"self_reported={ud['self_reported_confidences']}")

    # ── Final summary ────────────────────────────────────────────────────
    print_banner("RESULTS SUMMARY")
    print(f"  {'Problem':<25} {'Expected':<12} {'Smart Conf':>11} {'Raw Conf':>10} {'Self Avg':>9} {'Majority':<15} {'Verdict'}")
    print(f"  {'─'*25} {'─'*12} {'─'*11} {'─'*10} {'─'*9} {'─'*15} {'─'*40}")
    for r in results_summary:
        print(f"  {r['problem']:<25} {r['expected']:<12} {r['smart_conf']:>11.2f} "
              f"{r['raw_conf']:>10.2f} {r['self_reported_avg']:>9.2f} "
              f"{str(r['majority'])[:15]:<15} {r['verdict']}")

    print_banner("DIAGNOSIS")
    has_extractor_gap = any(r["raw_conf"] < 0.5 and r["smart_conf"] >= 0.6 for r in results_summary)
    has_self_report_gap = any(abs(r["self_reported_avg"] - r["smart_conf"]) > 0.3 for r in results_summary)

    if has_extractor_gap:
        print("  🔴 CRITICAL: The default string comparator significantly underestimates")
        print("     confidence. All answers may be correct but phrased differently.")
        print("  → FIX: Pass an answer_extractor (e.g., extract \\boxed{}) when creating")
        print("     the UncertaintyEstimator or via ControllerConfig.answer_extractor.")
        print()

    if has_self_report_gap:
        print("  ⚠️  Self-reported confidence diverges from measured confidence.")
        print("  → This validates the need for sampling-based uncertainty estimation — ")
        print("     the model's self-reported confidence is not well-calibrated.")
        print()

    all_correct = all(r["correct"] for r in results_summary)
    all_high = all(r["smart_conf"] >= 0.6 for r in results_summary)
    if all_correct and all_high:
        print("  ✅ With the smart extractor, uncertainty scores are WELL-CALIBRATED:")
        print("     high confidence on problems with consistent correct answers.")
    elif all_correct and not all_high:
        print("  ⚠️  Answers are correct but confidence is still low — the extractor or")
        print("     comparator may need further tuning.")
    print()


if __name__ == "__main__":
    main()
