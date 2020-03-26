import pandas as pd

res_tree = pd.read_csv('Ensemble_Learning_Rakuten_Challenge/Results/res_grid_tree.csv')


def list_files(model):
    res = []
    for i in range(2,10):
        res += ['res_grid_{}_{}.csv'.format(model,i)]
    return res


models = ['tree', 'bagging', 'rf']

list_files_models = {}
for model in models:
    list_files_models[model] = list_files(model)


res_models = {}

for model in models:
    res_models[model] = pd.read_csv('Ensemble_Learning_Rakuten_Challenge/Results/res_grid_{}.csv'.format(model))
    for file in list_files_models[model]:
        try:
            res_models[model] = res_models[model].append(
                pd.read_csv('Ensemble_Learning_Rakuten_Challenge/Results/{}'.format(file))
            )
        except:
            pass
    res_models[model].to_csv('Ensemble_Learning_Rakuten_Challenge/Results/res_grid_{}_full.csv'.format(model))
    res_models[model][['mean_fit_time', 'params',
              'mean_test_score',
              'mean_train_score']].to_csv(
        'Ensemble_Learning_Rakuten_Challenge/Results/res_grid_{}_synthesis.csv'.format(model))


synthesis_tree = pd.read_csv('Ensemble_Learning_Rakuten_Challenge/Results/res_grid_tree_synthesis.csv')
synthesis_bagging = pd.read_csv('Ensemble_Learning_Rakuten_Challenge/Results/res_grid_bagging_synthesis.csv')
synthesis_rf = pd.read_csv('Ensemble_Learning_Rakuten_Challenge/Results/res_grid_rf_synthesis.csv')
