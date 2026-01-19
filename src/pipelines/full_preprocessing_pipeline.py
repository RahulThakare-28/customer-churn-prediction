
from src.data_cleaning.leakage_remover import LeakageRemover
from src.data_cleaning.missing_values import MissingValueHandler
from src.data_cleaning.outliers import OutlierCapper
from src.data_cleaning.type_casting import TypeCaster
from src.feature_engineering.feature_creation import FeatureCreator
from src.feature_engineering.categorical_encoding import DynamicColumnTransformer
from src.feature_selection.correlation_filter import CorrelationFilter
from src.feature_selection.importance_selector import importance_selector
from src.feature_selection.pca_extractor import pca_transformer
from sklearn.pipeline import Pipeline

def build_full_pipeline():
    pipeline = Pipeline(steps=[
        ("leakage_removal", LeakageRemover()),
        ('missing', MissingValueHandler(threshold=0.5, protected_cols=['Offer'])),
        ('type_cast', TypeCaster()),
        ('outliers', OutlierCapper()),
        ('feature_create', FeatureCreator()),

        # DataFrame-based steps must be BEFORE encoding
        ('corr_filter', CorrelationFilter()),

        # Encoding converts to numpy array
        ('encoding_scaling', DynamicColumnTransformer()),

        # Array-based steps ONLY after this
        ('feature_select', importance_selector()),
        ('pca', pca_transformer())
    ])

    return pipeline
