"""Data processing helpers for biofuels modeling datasets."""

import pandas as pd

from pathlib import Path
from typing import ClassVar


class DataProcessing:
    """Process raw substance datasets into UNIFAC-derived feature datasets."""

    GROUPS: ClassVar[dict[str, dict[str, int]]] = {
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

    def __init__(
        self,
        input_csv: str,
        output_csv: str,
        project_root: Path | str | None = None,
    ) -> None:
        """Load raw input data and UNIFAC parameters for processing."""
        self.project_root = (
            Path(project_root).resolve() if project_root else self._find_project_root()
        )
        self.input_data_path = self.project_root / "data" / "private" / "raw"
        self.unifac_data_path = (
            self.project_root / "data" / "public" / "models_parameters"
        )
        self.output_data_path = self.project_root / "data" / "private" / "processed"

        self.raw_df = pd.read_csv(self.input_data_path / input_csv)
        self.unifac_parameters = pd.read_csv(
            self.unifac_data_path / "unifac_r_and_q.csv"
        )
        self.output_csv = output_csv

    @staticmethod
    def _find_project_root() -> Path:
        """Find the repository root from the installed package location."""
        for parent in Path(__file__).resolve().parents:
            if (parent / "data").exists() and (parent / "src").exists():
                return parent

        return Path.cwd().resolve()

    def __calculate_r(self, substance: str) -> float:
        """Calculate the UNIFAC R volume value for one substance."""
        unifac_r_values = self.unifac_parameters.set_index("Subgroup")[
            "R (volume)"
        ].to_dict()

        substance_group = self.GROUPS.get(substance)
        if substance_group is None:
            return 0.0

        initial_r = sum(
            group_count * unifac_r_values[group_name]
            for group_name, group_count in substance_group.items()
        )
        return round(initial_r, 4)

    def __calculate_q(self, substance: str) -> float:
        """Calculate the UNIFAC Q area value for one substance."""
        unifac_q_values = self.unifac_parameters.set_index("Subgroup")[
            "Q (area)"
        ].to_dict()

        substance_group = self.GROUPS.get(substance)
        if substance_group is None:
            return 0.0

        initial_q = sum(
            group_count * unifac_q_values[group_name]
            for group_name, group_count in substance_group.items()
        )
        return round(initial_q, 4)

    def process_raw_dataset(self) -> None:
        """Create R and Q columns, reorder the dataset, and save it as CSV."""
        substance_columns = [
            column for column in self.raw_df.columns if column.startswith("substance")
        ]

        for index, column in enumerate(substance_columns, start=1):
            self.raw_df[[f"r_{index}", f"q_{index}"]] = self.raw_df[column].apply(
                lambda substance: pd.Series(
                    [self.__calculate_r(substance), self.__calculate_q(substance)]
                )
            )

        if "FP" not in self.raw_df.columns:
            self.raw_df.drop(columns=substance_columns, inplace=True)

        num_substances = len(substance_columns)
        new_order = [
            column
            for index in range(1, num_substances + 1)
            for column in (f"r_{index}", f"q_{index}", f"x_{index}")
        ]

        if "FP" in self.raw_df.columns:
            new_order += ["MM", "lnPvap", "Method", "substance_1", "substance_2", "FP"]
        else:
            new_order += ["T"] + [
                f"ln_gamma_{index}" for index in range(1, num_substances + 1)
            ]

        self.output_data_path.mkdir(parents=True, exist_ok=True)
        self.raw_df[new_order].to_csv(
            self.output_data_path / self.output_csv, index=False
        )
