
GENOTYPE_ALIASES = {'WT' : ['WT', 'wildtype'],
                        'KO' : ['KO', 'knockout']}
CHNAME_ALIASES = {'A' : ['Aud', 'aud'],
                    'V' : ['Vis', 'vis'],
                    'H' : ['Hip', 'hip'],
                    'B' : ['Bar', 'bar'],
                    'M' : ['Mot', 'mot'],
                    # 'S' : ['Som', 'som']
                    }
LR_ALIASES = {'L' : ['left', 'Left', 'L ', ' L'],
            'R' : ['right', 'Right', 'R ', ' R']}

DEFAULT_CHNUM_TO_NAME = {9: 'LA',
                        10: 'LV',
                        12: 'LH',
                        14: 'LB',
                        15: 'LM',
                        16: 'RM',
                        17: 'RB',
                        19: 'RH',
                        21: 'RV',
                        22: 'RA',}
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
