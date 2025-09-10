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
        "ethyl octanoate": {"CH3": 2, "CH2": 7, "COO": 1},
        "ethyl decanoate": {"CH3": 2, "CH2": 9, "COO": 1},
        "ethyl laurate": {"CH3": 2, "CH2": 11, "COO": 1},
        "ethyl linoleate": {"CH3": 2, "CH2": 13, "CH=CH": 2, "COO": 1},
        "ethyl myristate": {"CH3": 2, "CH2": 13, "COO": 1},
        "ethyl oleate": {"CH3": 2, "CH2": 15, "CH=CH": 1, "COO": 1},
        "ethyl palmitate": {"CH3": 2, "CH2": 15, "COO": 1},
        "ethyl stearate": {"CH3": 2, "CH2": 17, "COO": 1},
        "ethylbenzene": {"ACH": 5, "ACCH2": 1, "CH3": 1},
        "ethyl propionate": {"CH3": 2, "CH2": 2, "COO": 1},
        "heptane": {"CH3": 2, "CH2": 5},
        "hexane": {"CH3": 2, "CH2": 4},
        "isobutanol": {"CH3": 2, "CH": 1, "CH2": 1, "OH": 1},
        "isopropanol": {"CH3": 2, "CH": 1, "OH": 1},
        "methanol": {"CH3OH": 1},
        "methyl acetate": {"CH3": 2, "COO": 1},
        "methyl butyrate": {"CH3": 2, "CH2": 2, "COO": 1},
        "methyl decanoate": {"CH3": 2, "CH2": 8, "COO": 1},
        "methyl laurate": {"CH3": 2, "CH2": 10, "COO": 1},
        "methyl linoleate": {"CH3": 2, "CH2": 12, "CH=CH": 2, "COO": 1},
        "methyl myristate": {"CH3": 2, "CH2": 12, "COO": 1},
        "methyl octanoate": {"CH3": 2, "CH2": 6, "COO": 1},
        "methyl oleate": {"CH3": 2, "CH2": 14, "CH=CH": 1, "COO": 1},
        "methyl palmitate": {"CH3": 2, "CH2": 14, "COO": 1},
        "methyl propionate": {"CH3": 2, "CH2": 1, "COO": 1},
        "methyl stearate": {"CH3": 2, "CH2": 16, "COO": 1},
        "nonane": {"CH3": 2, "CH2": 7},
        "octane": {"CH3": 2, "CH2": 6},
        "propanol": {"CH3": 1, "CH2": 2, "OH": 1},
        "propyl acetate": {"CH3": 2, "CH2": 2, "COO": 1},
        "toluene": {"AC": 5, "ACCH3": 1},
        "undecane": {"CH3": 2, "CH2": 9},
    }
    # Define the data path
    INPUT_DATA_PATH = Path("../data/private/raw")
    UNIFAC_DATA_PATH = Path("../data/public/models_parameters")
    OUTPUT_DATA_PATH = Path("../data/private/processed")

    def __init__(self, input_csv: str, output_csv: str) -> None:
        # Read csv file
        self.raw_df = pd.read_csv(self.INPUT_DATA_PATH / input_csv)

        # Load UNIFAC parameters
        self.unifac_parameters = pd.read_csv(
            self.UNIFAC_DATA_PATH / "unifac_r_and_q.csv"
        )
        # Initialize output csv path
        self.output_csv = output_csv

    def __calculate_r(self, substance: str) -> float:
        # Get UNIFAC R values
        unifac_r_values = self.unifac_parameters.set_index("Subgroup")[
            "R (volume)"
        ].to_dict()

        substance_group = self.GROUPS.get(substance, None)
        if substance_group is None:
            return 0.0

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
            return 0.0

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

        for i, col in enumerate(substance_columns, start=1):
            self.raw_df[[f"r_{i}", f"q_{i}"]] = self.raw_df[col].apply(
                lambda val: pd.Series(
                    [self.__calculate_r(val), self.__calculate_q(val)]
                )
            )

        # Drop substance columns
        if "FP" not in self.raw_df.columns:
            self.raw_df.drop(columns=substance_columns, inplace=True)

        # Reorder the columns
        num_substances = len(substance_columns)
        new_order = [
            col
            for i in range(1, num_substances + 1)
            for col in (f"r_{i}", f"q_{i}", f"x_{i}")
        ]

        if "FP" in self.raw_df.columns:
            new_order += ["MM", "lnPvap", "Method", "substance_1", "substance_2", "FP"]
        else:
            new_order += ["T"] + [f"ln_gamma_{i}" for i in range(1, num_substances + 1)]

        # Save table
        self.raw_df[new_order].to_csv(
            self.OUTPUT_DATA_PATH / self.output_csv, index=False
        )
