
import pandas as pd

from ... import visualization as viz


class ExperimentPlotter(viz.AnimalFeatureParser):
    def __init__(self, wars:list[viz.WindowAnalysisResult], features=['all'], exclude=None, groupby=['animalday', 'animal', 'genotype']) -> None:
        self.results: list[viz.WindowAnalysisResult] = wars
        self.channel_names: list[list[str]] = [war.channel_names for war in wars]
        dftemp = []
        for i, war in enumerate(wars):
            df = war.get_groupavg_result(features=features, exclude=exclude, groupby=groupby)
            df.insert(-1, 'chnames', self.channel_names[i])
        self.df_results:pd.DataFrame = pd.concat(dftemp, axis=1)

    def plot_boxplots(self, ):
        ...

    # flatten each column down to scalars, for ease of plotting. Or extract to multiindex column?
    # Try proof of concept with a test boxplot
    def _get_columnflattened_results(self):
        ...