"""
Class designed to process the raw dataset containing the name of substances, to
a dataset with r and q values from UNIFAC.

"""

import pandas as pd

from pathlib import Path


class DataProcessing:
    # Number of groups per substances
    GROUPS = {
        "benzene": {"ACH": 6},
        "butanol": {"CH3": 1, "CH2": 3, "OH": 1},
        "cyclohexane": {"CH2": 6},
        "cyclooctane": {"CH2": 8},
        "decane": {"CH3": 2, "CH2": 8},
        "dodecane": {"CH3": 2, "CH2": 10},
        "ethanol": {"CH3": 1, "CH2": 1, "OH": 1},
        "ethyl acetate": {"CH3": 2, "CH2": 1, "COO": 1},
        "ethylbenzene": {"ACH": 5, "ACCH2": 1, "CH3": 1},
        "ethyl propionate": {"CH3": 2, "CH2": 2, "COO": 1},
        "heptane": {"CH3": 2, "CH2": 5},
        "hexane": {"CH3": 2, "CH2": 4},
        "isobutanol": {"CH3": 2, "CH": 1, "CH2": 1, "OH": 1},
        "isopropanol": {"CH3": 2, "CH": 1, "OH": 1},
        "methanol": {"CH3OH": 1},
        "methyl acetate": {"CH3": 2, "COO": 1},
        "methyl laurate": {"CH3": 2, "CH2": 10, "COO": 1},
        "methyl linoleate": {"CH3": 2, "CH2": 12, "CH=CH": 2, "COO": 1},
        "methyl myristate": {"CH3": 2, "CH2": 12, "COO": 1},
        "methyl oleate": {"CH3": 2, "CH2": 14, "CH=CH": 1, "COO": 1},
        "methyl palmitate": {"CH3": 2, "CH2": 14, "COO": 1},
        "methyl propionate": {"CH3": 2, "CH2": 1, "COO": 1},
        "methyl stearate": {"CH3": 2, "CH2": 16, "COO": 1},
        "nonane": {"CH3": 2, "CH2": 7},
        "octane": {"CH3": 2, "CH2": 6},
        "propanol": {"CH3": 1, "CH2": 2, "OH": 1},
        "toluene": {"AC": 5, "ACCH3": 1},
        "undecane": {"CH3": 2, "CH2": 9},
    }

    def __init__(self) -> None:
        # Define the data path
        DATA_PATH = Path("../data")

        # Read csv file
        self.raw_df = pd.read_csv(DATA_PATH / "raw" / "toy_problem_raw_dataset.csv")

        # Load UNIFAC parameters
        self.unifac_parameters = pd.read_csv(
            DATA_PATH / "unifac_parameters" / "unifac_r_and_q.csv"
        )

    def __calculate_r(self, substance: str) -> float:
        # Get UNIFAC R values
        unifac_r_values = self.unifac_parameters.set_index("Subgroup")[
            "R (volume)"
        ].to_dict()

        substance_group = self.GROUPS.get(substance, None)
        if substance_group is None:
            raise ValueError(f"Substance {substance} not found in UNIFAC groups.")

        # Calculate r value
        initial_r = 0
        for key in substance_group.keys():
            initial_r += substance_group[key] * unifac_r_values[key]

        return round(initial_r, 4)

    def __calculate_q(self, substance: str) -> float:
        # Get UNIFAC R values
        unifac_q_values = self.unifac_parameters.set_index("Subgroup")[
            "Q (area)"
        ].to_dict()

        substance_group = self.GROUPS.get(substance, None)
        if substance_group is None:
            raise ValueError(f"Substance {substance} not found in UNIFAC groups.")

        # Calculate r value
        initial_q = 0
        for key in substance_group.keys():
            initial_q += substance_group[key] * unifac_q_values[key]

        return round(initial_q, 4)

    def process_raw_dataset(self):
        # Create r and q columns
        substance_columns = [
            col for col in self.raw_df.columns if col.startswith("substance")
        ]

        for index, col in enumerate(substance_columns, start=1):
            self.raw_df[f"r_{index}"] = self.raw_df[col].apply(self.__calculate_r)
            self.raw_df[f"q_{index}"] = self.raw_df[col].apply(self.__calculate_q)

        # Drop substance columns
        self.raw_df.drop(substance_columns, axis=1, inplace=True)

        # Reorder the columns
        new_order = [
            col
            for i in range(1, len(substance_columns) + 1)
            for col in [f"r_{i}", f"q_{i}", f"x_{i}"]
        ]
        new_order += ["T"]
        new_order += [
            col for i in range(1, len(substance_columns) + 1) for col in [f"gamma_{i}"]
        ]

        processed_df = self.raw_df[new_order]

        # Save table
        processed_df.to_csv(
            "../data/processed/toy_problem_input_dataset.csv", index=False
        )
