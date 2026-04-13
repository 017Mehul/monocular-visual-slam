## Contributing

Thanks for wanting to contribute. This file describes the minimal workflow
to run tests, linters, and add improvements.

1. Code style

 - We use `ruff` for linting and `black` for formatting. Run:

```bash
pip install -r requirements-dev.txt
ruff check .
black --check .
```

2. Tests

 - Run the pytest suite:

```bash
pytest -q
```

3. Adding benchmarks

 - The `benchmarks/run_kitti_bench.py` script aggregates ATE for sequences.
 - Producing per-sequence `trajectory_<seq>.csv` files in a folder and pointing
   the script at that folder will generate a CSV summary.

4. Sensor fusion

 - A minimal IMU loader stub is available at `slam/imu_interface.py`.
 - Use this file as the starting point for preintegration / bias estimation.

5. Pull requests

 - Open a PR against `main` and include a short description of changes.
 - Add tests for new functionality where practical.
