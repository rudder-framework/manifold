"""
PRISM API Server
================

Receives data from ORTHON, runs analysis, returns results.

Run:
    prism-serve
    # or
    python -m prism.api

Endpoints:
    GET  /health     - Health check
    POST /analyze    - Run analysis, return JSON
    POST /analyze/stream - Run analysis with streaming progress
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import json
from pathlib import Path
from typing import Optional

app = FastAPI(
    title="PRISM",
    description="Pure calculation engine for industrial diagnostics",
    version="0.1.0",
)

# Allow ORTHON to call from different origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    """Health check endpoint."""
    from prism import __version__
    return {
        "status": "ok",
        "engine": "prism",
        "version": __version__,
    }


@app.get("/engines")
async def list_engines():
    """List available engines by category."""
    from prism.capability import get_available_engines
    return get_available_engines()


@app.post("/analyze")
async def analyze(
    data: UploadFile = File(...),
    config: str = Form(default="{}"),
):
    """
    Run PRISM analysis on uploaded data.

    Args:
        data: CSV or Parquet file
        config: JSON string with analysis config
            - entity_column: Column identifying entities
            - constants: Dict of physical constants
            - domain: Domain hint (turbomachinery, fluid, etc.)

    Returns:
        Analysis results as JSON with paths to output files
    """
    import polars as pl

    # Parse config
    try:
        cfg = json.loads(config)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON in config")

    # Save uploaded file
    suffix = Path(data.filename).suffix.lower() if data.filename else '.parquet'
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await data.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        # Load data
        if suffix == '.parquet':
            df = pl.read_parquet(tmp_path)
        elif suffix in ('.csv', '.tsv', '.txt'):
            sep = '\t' if suffix == '.tsv' else ','
            df = pl.read_csv(tmp_path, separator=sep)
        else:
            df = pl.read_csv(tmp_path)

        # Run PRISM analysis
        from prism import analyze as prism_analyze
        results = prism_analyze(df, config=cfg)

        # Return summary (not full DataFrames)
        return {
            "status": "ok",
            "summary": results.summary(),
            "paths": {
                "data": str(results.data_path),
                "vector": str(results.vector_path),
                "geometry": str(results.geometry_path),
                "dynamics": str(results.dynamics_path),
                "physics": str(results.physics_path),
            },
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        tmp_path.unlink(missing_ok=True)


@app.post("/analyze/dataframe")
async def analyze_dataframe(
    data: UploadFile = File(...),
    config: str = Form(default="{}"),
    output: str = Form(default="vector"),
):
    """
    Run analysis and return a specific output as Parquet.

    Args:
        data: CSV or Parquet file
        config: JSON config string
        output: Which output to return (data, vector, geometry, dynamics, physics)

    Returns:
        Parquet file as streaming response
    """
    import polars as pl
    import io

    cfg = json.loads(config) if config else {}

    suffix = Path(data.filename).suffix.lower() if data.filename else '.parquet'
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await data.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        if suffix == '.parquet':
            df = pl.read_parquet(tmp_path)
        else:
            df = pl.read_csv(tmp_path)

        from prism import analyze as prism_analyze
        results = prism_analyze(df, config=cfg)

        # Get requested output
        output_map = {
            'data': results.data,
            'vector': results.vector,
            'geometry': results.geometry,
            'dynamics': results.dynamics,
            'physics': results.physics,
        }

        if output not in output_map:
            raise HTTPException(status_code=400, detail=f"Invalid output: {output}")

        result_df = output_map[output]

        # Write to bytes
        buffer = io.BytesIO()
        result_df.write_parquet(buffer)
        buffer.seek(0)

        return StreamingResponse(
            buffer,
            media_type="application/octet-stream",
            headers={"Content-Disposition": f"attachment; filename={output}.parquet"},
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        tmp_path.unlink(missing_ok=True)


@app.post("/intake")
async def intake_only(
    data: UploadFile = File(...),
    config: str = Form(default="{}"),
):
    """
    Run intake only (unit detection, validation) without full analysis.

    Useful for quick data inspection.
    """
    import polars as pl

    cfg = json.loads(config) if config else {}

    suffix = Path(data.filename).suffix.lower() if data.filename else '.parquet'
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await data.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        if suffix == '.parquet':
            df = pl.read_parquet(tmp_path)
        else:
            df = pl.read_csv(tmp_path)

        from prism.intake import ingest
        result = ingest(df, config=cfg)

        return {
            "status": "ok",
            "level": result.level,
            "entity_column": result.entity_column,
            "time_column": result.time_column,
            "signals": [
                {"name": s.name, "unit": s.unit, "category": s.category}
                for s in result.signals
            ],
            "units": result.units,
            "issues": result.issues,
            "rows": len(df),
            "columns": len(df.columns),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        tmp_path.unlink(missing_ok=True)


def main():
    """Run the PRISM API server."""
    import uvicorn
    uvicorn.run(
        "prism.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )


if __name__ == "__main__":
    main()
