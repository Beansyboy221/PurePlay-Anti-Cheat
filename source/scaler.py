import polars
import numpy

class PolarsStandardScaler:
    """Custom standard scaler using Polars for efficient computation."""
    def __init__(self):
        self.mean = None
        self.standard_deviation = None
        self.columns = []

    def fit(self, file_paths, scalable_columns):
        if not scalable_columns:
            return

        statistics = polars.scan_parquet(file_paths).select([
            *[polars.col(column).mean().alias(f"{column}_mean") for column in scalable_columns],
            *[polars.col(column).std().alias(f"{column}_standard_deviation") for column in scalable_columns]
        ]).collect()

        self.columns = scalable_columns
        self.mean = numpy.array([statistics.get_column(f"{column}_mean")[0] for column in scalable_columns], dtype=numpy.float32)
        self.standard_deviation = numpy.array([statistics.get_column(f"{column}_standard_deviation")[0] for column in scalable_columns], dtype=numpy.float32)
        
        self.standard_deviation[self.standard_deviation == 0] = 1.0

    def transform(self, data):
        return (data - self.mean) / self.standard_deviation

SCALER = PolarsStandardScaler()