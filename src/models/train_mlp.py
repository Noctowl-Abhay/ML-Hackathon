from sklearn.neural_network import MLPClassifier
try:
    from .common import run
except ImportError:
    from common import run

if __name__ == '__main__':
    run('deep_mlp',MLPClassifier(hidden_layer_sizes=(128,64,32),activation='relu',alpha=0.001,early_stopping=True,validation_fraction=0.15,max_iter=150,batch_size=256,random_state=42),scale=True)
