from pathlib import Path

import pandas as pd

from energy_mlops.data.schema import CapacityCheckpointSchema

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_CAPACITY_CHECKPOINTS_PATH = (
    PROJECT_ROOT
    / "data"
    / "reference"
    / "bahia_wind_capacity_checkpoints.csv"
)


def load_bahia_wind_capacity_checkpoints(
    path: str | Path = DEFAULT_CAPACITY_CHECKPOINTS_PATH,
) -> pd.DataFrame:
    """
    Carrega os checkpoints de capacidade eólica da Bahia.

    A referência canônica é construída a partir dos boletins
    INFOVENTO da ABEEólica.

    As três datas possuem semânticas distintas:

    - reference_date:
      data/período histórico ao qual a capacidade é atribuída;

    - publication_date:
      data de publicação do boletim;

    - available_from:
      data a partir da qual o pipeline pode utilizar
      causalmente o checkpoint.

    O DataFrame retornado é validado por
    CapacityCheckpointSchema e ordenado por available_from.
    """

    capacity_path = Path(path)

    if not capacity_path.is_file():
        raise FileNotFoundError(
            "Arquivo de checkpoints de capacidade não encontrado: "
            f"{capacity_path}"
        )

    df = pd.read_csv(
        capacity_path,
        parse_dates=[
            "publication_date",
            "reference_date",
            "available_from",
        ],
    )

    validated = CapacityCheckpointSchema.validate(df)

    validated = (
        validated
        .sort_values("available_from")
        .reset_index(drop=True)
    )

    return validated