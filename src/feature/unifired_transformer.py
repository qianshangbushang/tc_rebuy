import os
import pickle

from sklearn.base import BaseEstimator, TransformerMixin

from .behavior_transformer import BehaviorTransformer
from .matrix_transformer import MatrixTransformer
from .tfidf_transformer import TfidfTransformer


class UnifiedFeatureTransformer(BaseEstimator, TransformerMixin):
    """统一的特征转换器，整合多个特征工程步骤"""

    def __init__(
        self,
        # 行为特征配置
        enable_behavior: bool = True,
        behavior_cache_path: str = "../data/behavior_feature.pkl",
        # TF-IDF特征配置
        enable_tfidf: bool = True,
        tfidf_top_n_features: int = 50,
        tfidf_cache_path: str = "../data/tfidf_feature.pkl",
        # 矩阵分解特征配置
        enable_matrix: bool = True,
        matrix_n_components: int = 20,
        matrix_cache_path: str = "../data/matrix_feature.pkl",
        # 全局配置
        enable_cache: bool = True,
        unified_cache_path: str = "../data/unified_features.pkl",
        random_state: int = 42,
    ):
        """
        Args:
            enable_behavior: 是否启用行为特征
            enable_tfidf: 是否启用TF-IDF特征
            enable_matrix: 是否启用矩阵分解特征
            enable_cache: 是否启用统一缓存
            unified_cache_path: 统一缓存路径
        """
        self.enable_behavior = enable_behavior
        self.enable_tfidf = enable_tfidf
        self.enable_matrix = enable_matrix
        self.enable_cache = enable_cache
        self.unified_cache_path = unified_cache_path

        # 初始化各个transformer
        self.transformers = {}

        if enable_behavior:
            self.transformers["behavior"] = BehaviorTransformer(
                enable_cache=enable_cache, cache_path=behavior_cache_path
            )

        if enable_tfidf:
            self.transformers["tfidf"] = TfidfTransformer(
                top_n_features=tfidf_top_n_features, enable_cache=enable_cache, cache_path=tfidf_cache_path
            )

        if enable_matrix:
            self.transformers["matrix"] = MatrixTransformer(
                n_components=matrix_n_components,
                random_state=random_state,
                enable_cache=enable_cache,
                cache_path=matrix_cache_path,
            )

        self.is_fitted = False
        self.feature_names_ = []

    def fit(self, X, y=None):
        """训练所有特征转换器"""

        # 检查统一缓存
        if self.enable_cache and os.path.exists(self.unified_cache_path):
            print(f"♻️ 从统一缓存加载特征: {self.unified_cache_path}")
            with open(self.unified_cache_path, "rb") as f:
                cached_data = pickle.load(f)

            # 加载各个transformer的状态
            for name, transformer in self.transformers.items():
                if name in cached_data:
                    transformer.__dict__.update(cached_data[name])

            self.feature_names_ = cached_data.get("feature_names", [])
            print("✅ 统一特征加载完成")
            self.is_fitted = True
            return self

        print("🔄 开始统一特征训练...")

        # 依次训练各个transformer
        for name, transformer in self.transformers.items():
            print(f"  🔄 训练{name}特征...")
            transformer.fit(X, y)
            print(f"  ✅ {name}特征训练完成")

        # 收集特征名
        self._collect_feature_names()

        # 缓存统一结果
        if self.enable_cache:
            self._cache_unified_features()

        self.is_fitted = True
        print("✅ 统一特征训练完成")
        return self

    def transform(self, X):
        """应用所有特征转换"""
        if not self.is_fitted:
            raise ValueError("UnifiedFeatureTransformer has not been fitted yet.")

        print("🔄 开始统一特征转换...")

        result = X.copy()

        # 依次应用各个transformer
        for name, transformer in self.transformers.items():
            print(f"  🔄 应用{name}特征转换...")
            result = transformer.transform(result)
            print(f"  ✅ {name}特征转换完成: {result.shape}")

        print(f"✅ 统一特征转换完成，最终形状: {result.shape}")
        return result

    def _collect_feature_names(self):
        """收集所有特征名"""
        self.feature_names_ = []

        for name, transformer in self.transformers.items():
            if hasattr(transformer, "get_feature_names"):
                try:
                    names = transformer.get_feature_names()
                    self.feature_names_.extend(names)
                except:
                    print(f"⚠️ 无法获取{name}的特征名")

    def _cache_unified_features(self):
        """缓存统一特征"""
        print("💾 缓存统一特征...")
        os.makedirs(os.path.dirname(self.unified_cache_path), exist_ok=True)

        cached_data = {"feature_names": self.feature_names_}

        # 保存各个transformer的状态
        for name, transformer in self.transformers.items():
            cached_data[name] = transformer.__dict__.copy()

        with open(self.unified_cache_path, "wb") as f:
            pickle.dump(cached_data, f)
        print(f"✅ 统一特征已缓存到: {self.unified_cache_path}")

    def get_feature_names(self):
        """获取所有特征名"""
        if not self.is_fitted:
            raise ValueError("Transformer has not been fitted yet.")
        return self.feature_names_

    def get_feature_stats(self):
        """获取特征统计信息"""
        if not self.is_fitted:
            return None

        stats = {}
        for name, transformer in self.transformers.items():
            if hasattr(transformer, "get_embedding_stats"):
                stats[name] = transformer.get_embedding_stats()
            elif hasattr(transformer, "get_feature_names"):
                try:
                    feature_count = len(transformer.get_feature_names())
                    stats[name] = {"feature_count": feature_count}
                except:
                    stats[name] = {"status": "fitted"}

        return stats
