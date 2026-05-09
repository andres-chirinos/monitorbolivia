#!/usr/bin/env python3
"""ETL + Kaggle publish pipeline for monitorbolivia."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import yaml


def load_env_from_dotenv(repo_root: Path) -> Path | None:
    dotenv_candidates = [repo_root / ".env", Path.cwd() / ".env"]
    dotenv_path = next((path for path in dotenv_candidates if path.exists()), None)
    if dotenv_path is None:
        return None

    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if line.startswith("export "):
            line = line[len("export ") :].strip()

        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue

        if (
            len(value) >= 2
            and ((value[0] == '"' and value[-1] == '"') or (value[0] == "'" and value[-1] == "'"))
        ):
            value = value[1:-1]

        # Do not override values already provided by the environment.
        os.environ.setdefault(key, value)

    return dotenv_path


def _normalize_env_value(value: str | None) -> str:
    if value is None:
        return ""

    normalized = value.strip()
    if (
        len(normalized) >= 2
        and (
            (normalized[0] == '"' and normalized[-1] == '"')
            or (normalized[0] == "'" and normalized[-1] == "'")
        )
    ):
        normalized = normalized[1:-1].strip()
    return normalized


def resolve_kaggle_credentials() -> tuple[str, str, str]:
    username = _normalize_env_value(os.getenv("KAGGLE_USERNAME"))
    key = _normalize_env_value(os.getenv("KAGGLE_KEY"))
    api_token = _normalize_env_value(os.getenv("KAGGLE_API_TOKEN"))

    # Many users place a Kaggle API token (KGAT_...) in KAGGLE_KEY.
    # If present, prefer kagglehub auth via KAGGLE_API_TOKEN.
    if not api_token and key.startswith("KGAT_"):
        api_token = key

    if api_token:
        os.environ["KAGGLE_API_TOKEN"] = api_token
        return username, key, api_token

    missing = [
        name
        for name, value in (("KAGGLE_USERNAME", username), ("KAGGLE_KEY", key))
        if not value
    ]
    if missing:
        raise EnvironmentError(
            "Faltan credenciales para Kaggle. Define KAGGLE_API_TOKEN o "
            f"el par legacy ({', '.join(missing)})."
        )

    return username, key, api_token


def _iso_to_date(value: object) -> str | None:
    if not value:
        return None

    try:
        if isinstance(value, str):
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        elif isinstance(value, datetime):
            parsed = value
        else:
            return None
        return parsed.date().isoformat()
    except Exception:
        return None


def build_summary_dataframe(summary_json_path: Path, history_dir: Path) -> pd.DataFrame:
    with summary_json_path.open(encoding="utf-8") as file:
        history_items = json.load(file)

    down_frames: list[pd.DataFrame] = []
    for item in history_items:
        daily_minutes = item.get("dailyMinutesDown", {})
        if not isinstance(daily_minutes, dict) or not daily_minutes:
            continue

        temp_df = pd.DataFrame(daily_minutes.items(), columns=["date", "time"])
        temp_df["domain"] = urlparse(item.get("url", "")).netloc
        temp_df["type"] = "down"
        down_frames.append(temp_df)

    base_df = (
        pd.concat(down_frames, ignore_index=True)
        if down_frames
        else pd.DataFrame(columns=["date", "time", "domain", "type"])
    )

    records: list[dict[str, object]] = []
    for yml_file in sorted(history_dir.glob("*.yml")):
        with yml_file.open(encoding="utf-8") as file:
            item = yaml.safe_load(file) or {}

        domain = urlparse(str(item.get("url", ""))).netloc
        creation_date = _iso_to_date(item.get("startTime"))
        last_update = _iso_to_date(item.get("lastUpdated"))

        if creation_date:
            records.append(
                {
                    "domain": domain,
                    "date": creation_date,
                    "time": 0,
                    "type": "created",
                }
            )
        if last_update:
            records.append(
                {
                    "domain": domain,
                    "date": last_update,
                    "time": 0,
                    "type": "last_updated",
                }
            )

    metadata_df = pd.DataFrame(records)
    final_df = (
        pd.concat([base_df, metadata_df], ignore_index=True)
        if not metadata_df.empty
        else base_df
    )

    final_df["date"] = pd.to_datetime(final_df["date"], errors="coerce").dt.date
    final_df = final_df[final_df["date"].notna()].copy()

    final_df["time"] = pd.to_numeric(final_df["time"], errors="coerce").fillna(0).astype(int)
    final_df["domain"] = final_df["domain"].astype(str)
    final_df["type"] = final_df["type"].astype("category")

    return final_df


def ensure_kaggle_json(username: str, key: str) -> None:
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(parents=True, exist_ok=True)

    kaggle_json = kaggle_dir / "kaggle.json"
    kaggle_json.write_text(
        json.dumps({"username": username, "key": key}),
        encoding="utf-8",
    )
    kaggle_json.chmod(0o600)


def load_metadata(repo_root: Path, kaggle_id_override: str | None) -> dict[str, object]:
    metadata_candidates = [
        repo_root / "dataset-metadata.json",
        Path.cwd() / "dataset-metadata.json",
    ]

    source = next((p for p in metadata_candidates if p.exists()), None)
    if source is None:
        raise FileNotFoundError(
            "No se encontro dataset-metadata.json en la raiz del repo o cwd."
        )

    with source.open(encoding="utf-8") as file:
        metadata = json.load(file)

    if kaggle_id_override:
        metadata["id"] = kaggle_id_override

    if "id" not in metadata or not isinstance(metadata["id"], str):
        raise ValueError("dataset-metadata.json debe incluir un campo string 'id'.")

    return metadata


def publish_to_kaggle(
    data_dir: Path,
    metadata: dict[str, object],
    username: str,
    version_message: str,
) -> None:
    try:
        import kagglehub
    except ImportError as exc:
        raise RuntimeError(
            "No se encontro el paquete 'kagglehub'. Instala dependencias con: "
            "pip install -r pipeline/requirements.txt"
        ) from exc

    kaggle_id = str(metadata["id"])
    owner = kaggle_id.split("/", 1)[0] if "/" in kaggle_id else ""

    if username and owner and owner.lower() != username.lower():
        raise PermissionError(
            "El owner del dataset no coincide con KAGGLE_USERNAME. "
            f"owner en metadata: '{owner}', usuario actual: '{username}'."
        )

    print(f"Publicando dataset con kagglehub: {kaggle_id}")
    with tempfile.TemporaryDirectory(prefix="kagglehub-upload-") as temp_upload_dir:
        upload_dir = Path(temp_upload_dir)

        for item in data_dir.iterdir():
            destination = upload_dir / item.name
            if item.is_dir():
                shutil.copytree(item, destination)
            else:
                shutil.copy2(item, destination)

        #metadata_path = upload_dir / "dataset-metadata.json"
        #metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        try:
            kagglehub.dataset_upload(
                kaggle_id,
                str(upload_dir),
                version_notes=version_message,
            )
        except Exception as exc:
            error_text = str(exc).lower()
            if "403" in error_text or "forbidden" in error_text:
                raise PermissionError(
                    "Kaggle devolvio 403 Forbidden durante dataset_upload con kagglehub. "
                    f"dataset='{kaggle_id}', usuario='{username or 'n/a'}'. "
                    "Revisa autenticacion (KAGGLE_API_TOKEN o credenciales legacy), "
                    "owner del dataset y permisos."
                ) from exc
            raise RuntimeError(
                "Fallo kagglehub.dataset_upload. "
                f"Detalle original: {exc}"
            ) from exc


def parse_args() -> argparse.Namespace:
    repo_root_default = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(description="Build summary parquet and publish to Kaggle.")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=repo_root_default,
        help="Ruta raiz del repositorio.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=repo_root_default / "history" / "summary.json",
        help="Ruta al history/summary.json.",
    )
    parser.add_argument(
        "--history-dir",
        type=Path,
        default=repo_root_default / "history",
        help="Ruta al directorio con archivos .yml historicos.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=repo_root_default / "data",
        help="Directorio de salida para parquet y metadata de Kaggle.",
    )
    parser.add_argument(
        "--output-parquet",
        type=Path,
        default=None,
        help="Ruta de salida del parquet (default: <data-dir>/summary.parquet).",
    )
    parser.add_argument(
        "--kaggle-id",
        type=str,
        default=None,
        help="Sobrescribe metadata.id para publicar a otro dataset.",
    )
    parser.add_argument(
        "--version-message",
        type=str,
        default="daily update",
        help="Mensaje de version para Kaggle.",
    )
    parser.add_argument(
        "--skip-kaggle",
        action="store_true",
        help="Solo ejecuta ETL y genera parquet; no publica en Kaggle.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dotenv_path = load_env_from_dotenv(args.repo_root)
    if dotenv_path:
        print(f"Variables cargadas desde .env: {dotenv_path}")

    summary_json_path = args.summary_json
    history_dir = args.history_dir
    data_dir = args.data_dir
    output_parquet = args.output_parquet or (data_dir / "summary.parquet")

    print("Construyendo dataframe de resumen...")
    df = build_summary_dataframe(summary_json_path, history_dir)

    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_parquet, index=False, engine="pyarrow")
    print(f"Parquet generado: {output_parquet}")
    print(f"Filas: {len(df)}")

    if args.skip_kaggle:
        print("Se omite publicacion a Kaggle (--skip-kaggle).")
        return

    username, key, api_token = resolve_kaggle_credentials()
    if api_token:
        print("Autenticacion Kaggle: usando KAGGLE_API_TOKEN (kagglehub).")
    else:
        print("Autenticacion Kaggle: usando credenciales legacy (kaggle.json).")
        ensure_kaggle_json(username=username, key=key)

    metadata = load_metadata(repo_root=args.repo_root, kaggle_id_override=args.kaggle_id)
    print("metadata:", metadata)
    publish_to_kaggle(
        data_dir=data_dir,
        metadata=metadata,
        username=username,
        version_message=args.version_message,
    )


if __name__ == "__main__":
    main()
