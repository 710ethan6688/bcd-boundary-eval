"""Reproduction script for operational direction checks ($G_{min} > 0$) from paired count records."""
import argparse
import json
from pathlib import Path
from boundary_claim_acceptance import assess_boundary_claims

def main():
    parser = argparse.ArgumentParser(description="Assess directional consistency across declared conditions.")
    parser.add_argument('--input', type=Path, default=Path(__file__).parent / 'data' / 'LEVIR_4models_counts.json',
                        help="Path to paired model count JSON file.")
    parser.add_argument('--output', type=Path, default=Path(__file__).parent / 'output_assessment.json',
                        help="Path to save JSON evaluation report.")
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input count file not found: {args.input}")

    data = json.loads(args.input.read_text(encoding='utf-8'))
    results = assess_boundary_claims(data['frame_models'])

    print("=" * 80)
    print(f"{'Comparison':<30} | {'Conditions':<10} | {'G_min (pp)':<12} | {'Result':<10}")
    print("-" * 80)
    for comp, res in results.items():
        gmin_str = f"{res['minimum_gain_pp']:+.2f}" if res['minimum_gain_pp'] is not None else "N/A"
        status = "PASSED" if res['boundary_direction_passed'] else "FAILED"
        print(f"{comp:<30} | {res['paired_conditions']:<10} | {gmin_str:<12} | {status:<10}")
    print("=" * 80)

    output_payload = dict(
        scope=data.get('scope', 'declared LEVIR evaluation'),
        comparisons=results
    )
    args.output.write_text(json.dumps(output_payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"\nDetailed check results written to: {args.output}")

if __name__ == '__main__':
    main()
