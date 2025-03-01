DEFAULT_ID_TO_LR = {9 : 'L',
                    10 : 'L',
                    12 : 'L',
                    14 : 'L',
                    15 : 'L',
                    16 : 'R',
                    17 : 'R',
                    19 : 'R',
                    21 : 'R',
                    22 : 'R'}


GENOTYPE_ALIASES = {'WT' : ['WT', 'wildtype'],
                        'KO' : ['KO', 'knockout']}
CHNAME_ALIASES = {'Aud' : ['Aud', 'aud'],
                    'Vis' : ['Vis', 'vis'],
                    'Hip' : ['Hip', 'hip'],
                    'Bar' : ['Bar', 'bar'],
                    'Mot' : ['Mot', 'mot'],
                    # 'S' : ['Som', 'som']
                    }
LR_ALIASES = {'L' : ['left', 'Left', 'L ', ' L'],
            'R' : ['right', 'Right', 'R ', ' R']}
DEFAULT_ID_TO_NAME = {9: 'LAud',
                        10: 'LVis',
                        12: 'LHip',
                        14: 'LBar',
                        15: 'LMot',
                        16: 'RMot',
                        17: 'RBar',
                        19: 'RHip',
                        21: 'RVis',
                        22: 'RAud',}

FEATURES = ['rms', 'ampvar', 'psd', 'psdtotal', 'psdband', 'psdslope', 'cohere', 'pcorr', 'nspike', 'wavetemp']
LINEAR_FEATURE = ['rms', 'ampvar', 'psdtotal', 'psdslope']
BAND_FEATURE = ['psdband']
MATRIX_FEATURE = ['cohere', 'pcorr']
HIST_FEATURE = ['psd']
GLOBAL_FEATURES = ['templates']

FREQ_BANDS = {'delta' : (0.1, 4),
            'theta' : (4, 8),
            'alpha' : (8, 13),
            'beta'  : (13, 25),
            'gamma' : (25, 50)}
BAND_NAMES = [k for k,_ in FREQ_BANDS.items()]

FREQ_BAND_TOTAL = (0.1, 50)
FREQ_MINS = [v[0] for k,v in FREQ_BANDS.items()]
FREQ_MAXS = [v[1] for k,v in FREQ_BANDS.items()]
LINE_FREQ = 60