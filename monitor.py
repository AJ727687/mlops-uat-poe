import numpy as np
import pandas as pd
from evidently import Report, Dataset, DataDefinition
from evidently.presets import DataDriftPreset

rng = np.random.default_rng(42)
reference = pd.DataFrame({"feature_a": rng.normal(0, 1, 200),
                          "feature_b": rng.normal(10, 2, 200)})
current = pd.DataFrame({"feature_a": rng.normal(0.5, 1, 200),
                        "feature_b": rng.normal(10, 2, 200)})

definition = DataDefinition(numerical_columns=["feature_a", "feature_b"])
ref_ds = Dataset.from_pandas(reference, data_definition=definition)
cur_ds = Dataset.from_pandas(current, data_definition=definition)

report = Report([DataDriftPreset()])
result = report.run(cur_ds, ref_ds)
result.save_html("drift_report.html")
print("Evidently report generated: drift_report.html")