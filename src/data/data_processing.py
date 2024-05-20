""" 
Class designed to process the raw dataset containing the name of substances, to
a dataset with r and q values from UNIFAC.

"""

import pandas as pd

from ugropy import Groups


class DataProcessing:
    def __init__(self) -> None:
        # Read csv file
        self.raw_df = pd.read_csv("data/raw/dataset.csv")

        # Load UNIFAC parameters
        self.unifac_parameters = pd.read_csv(
            "data/unifac_parameters/unifac_r_and_q.csv"
        )

    def __calculate_r(self, substance: str) -> float:
        # Get UNIFAC R values
        unifac_r_values = self.unifac_parameters.set_index("Subgroup")[
            "R (volume)"
        ].to_dict()

        init_substance = Groups(substance)
        substance_group = init_substance.unifac.subgroups

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

        init_substance = Groups(substance)
        substance_group = init_substance.unifac.subgroups

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
        processed_df.to_csv("data/processed/input_dataset.csv", index=False)
