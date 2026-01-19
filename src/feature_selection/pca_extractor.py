from sklearn.decomposition import PCA

def pca_transformer(n_components=0.95):
    return PCA(n_components=n_components, random_state=42)