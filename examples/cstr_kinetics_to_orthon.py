"""
CSTR Kinetics → ORTHON Pipeline Demo

PRISM: Calculates pure numbers, outputs to physics.parquet with discipline tag
ORTHON: Reads discipline tag, applies discipline-specific formatting/reports

Key: PRISM outputs `discipline='reaction'` column. ORTHON knows what to do with it.
"""

import polars as pl
from pathlib import Path
import tempfile


def main():
    from prism.engines.physics.reaction_output import compute_reaction_to_orthon

    print("=" * 70)
    print("PRISM → ORTHON: Reaction Discipline")
    print("=" * 70)

    # Experimental data
    entity_ids = ['run_1', 'run_2', 'run_3', 'run_4', 'run_5']
    temperatures_K = [298.15, 308.15, 318.15, 328.15, 338.15]
    inlet_concentrations = [0.1] * 5
    outlet_concentrations = [0.0720, 0.0534, 0.0378, 0.0260, 0.0177]

    # Output path
    output_dir = Path(tempfile.mkdtemp(prefix='prism_'))
    output_path = output_dir / 'physics.parquet'

    print(f"\n[PRISM] Calculating reaction metrics...")

    # PRISM calculates and outputs
    result = compute_reaction_to_orthon(
        output_path=str(output_path),
        entity_ids=entity_ids,
        temperatures_K=temperatures_K,
        inlet_concentrations=inlet_concentrations,
        outlet_concentrations=outlet_concentrations,
        reactor_volume_L=2.5,
        flow_rate_mL_min=50,
        reaction_order=2,
        heat_of_reaction=-75300,
        pipe_diameter_m=0.0127,
        density_kg_m3=1020,
        viscosity_Pa_s=0.00102,
    )

    print(f"[PRISM] Output: {result['path']}")
    print(f"[PRISM] Discipline: {result['discipline']}")

    # Show what ORTHON receives
    print("\n" + "=" * 70)
    print("[ORTHON] Reading physics.parquet...")
    print("=" * 70)

    df = pl.read_parquet(result['path'])

    print(f"\nColumns: {df.columns}")
    print(f"\nDiscipline tag: {df['discipline'][0]}")

    print("\n📊 Data for ORTHON:")
    print(df.select([
        'entity_id', 'discipline', 'temperature_C', 'conversion',
        'rate_constant', 'heat_duty_W', 'reynolds',
        'activation_energy_kJ_mol', 'arrhenius_r_squared'
    ]))

    print(f"""
ORTHON sees discipline='reaction' and applies:
├── Labels: flow_regime='laminar' (Re={result['summary']['reynolds']:.0f})
├── Arrhenius plot: ln(k) vs 1/T
├── Material balance verification table
├── Energy balance heat duty chart
└── CHE 344 Lab Report template

PRISM output: {result['path']}
""")


if __name__ == '__main__':
    main()
