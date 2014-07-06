"""
"Standard datasets" follow a simple convention on UCI: they are located in a directory which includes a .data and a
.name file which makes them easy to fetch.
"""

standard_datasets = {}
standard_datast_base_url = "http://archive.ics.uci.edu/ml/machine-learning-databases/"

standard_datasets["abalone"] = "abalone/abalone"
standard_datasets["acute_inflamation"] = "/acute/diagnosis"
standard_datasets["arrhythmia"] = "arrhythmia/arrhythmia"
standard_datasets["badges"] = "bagdes/badges"
standard_datasets["balance_scale"] = "balance_scale/balance_scale"
standard_datasets["blood_transfusion"] = "blood-transfusion/transfusion"
standard_datasets["breast_cancer_wisconsin"] = "breast-cancer-wisconsin/breast-cancer-wisconsin"
standard_datasets["breast_cancer_wisconsin_prognostic"] = "breast-cancer-wisconsin/wppc"
standard_datasets["breast_cancer_wisconsin_diagnostic"] = "breast-cancer-wisconsin/wdbc"
standard_datasets["chess_king_rook_vs_king_pawn"] = "chess/king-rook-vs-king-pawn/kr-vs-kp"
standard_datasets["cnae_9"] = "00233/CNAE-9"
standard_datasets["congressional_voting_records_data_set"] = "voting-records/house-votes-84"
standard_datasets["contraceptive_method"] = "cmc/cmc"


"""
Pre-split datasets
"""
standard_datasets["adult"] = {"data": "http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
                     "data2": "http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test",
                     "meta": "http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names"}
standard_datasets["anneal"] = {"data": "http://archive.ics.uci.edu/ml/machine-learning-databases/annealing/anneal.data",
                      "data2": "http://archive.ics.uci.edu/ml/machine-learning-databases/annealing/anneal.test",
                      "meta": "http://archive.ics.uci.edu/ml/machine-learning-databases/annealing/anneal.names"}

standard_datasets["audiology"] = {"data": ["http://archive.ics.uci.edu/ml/machine-learning-databases/audiology/audiology.data",
                                  "http://archive.ics.uci.edu/ml/machine-learning-databases/audiology/audiology.test"],
                                 "meta": "http://archive.ics.uci.edu/ml/machine-learning-databases/audiology/audiology.names"}
standard_datasets["audiology_standardized"] = {"data": ["http://archive.ics.uci.edu/ml/machine-learning-databases/audiology/audiology.standardized.data",
                                                        "http://archive.ics.uci.edu/ml/machine-learning-databases/audiology/audiology.standardized.test"],
                                               "meta": "http://archive.ics.uci.edu/ml/machine-learning-databases/audiology/audiology.standardized.test"}


"""
Other datasets
"""
standard_datasets['arcene'] = {"data": ["http://archive.ics.uci.edu/ml/machine-learning-databases/arcene/ARCENE/arcene.param",
                               "http://archive.ics.uci.edu/ml/machine-learning-databases/arcene/ARCENE/arcene_test.data",
                               "http://archive.ics.uci.edu/ml/machine-learning-databases/arcene/ARCENE/arcene_train.data",
                               "http://archive.ics.uci.edu/ml/machine-learning-databases/arcene/ARCENE/arcene_train.labels",
                               "http://archive.ics.uci.edu/ml/machine-learning-databases/arcene/ARCENE/arcene_valid.data",
                               "http://archive.ics.uci.edu/ml/machine-learning-databases/arcene/arcene_valid.labels"],
                      "meta": "http://archive.ics.uci.edu/ml/machine-learning-databases/arcene/Dataset.pdf"}
standard_datasets["australian_language_signs"] = {"data": "http://archive.ics.uci.edu/ml/machine-learning-databases/auslan-mld/allsigns.tar.gz",
                                         "meta": "http://archive.ics.uci.edu/ml/machine-learning-databases/auslan-mld/auslan.data.html"}
standard_datasets["australian_language_signs_hq"] = {"data": "http://archive.ics.uci.edu/ml/machine-learning-databases/auslan2-mld/tctodd.tar.gz",
                                         "meta": "http://archive.ics.uci.edu/ml/machine-learning-databases/auslan2-mld/auslan.data.html"}
standard_datasets["balloons"] = {"data": ["http://archive.ics.uci.edu/ml/machine-learning-databases/balloons/adult+stretch.data",
                                          "http://archive.ics.uci.edu/ml/machine-learning-databases/balloons/adult-stretch.data",
                                          "http://archive.ics.uci.edu/ml/machine-learning-databases/balloons/yellow-small+adult-stretch.data",
                                          "http://archive.ics.uci.edu/ml/machine-learning-databases/balloons/yellow-small.data"],
                                 "meta": "http://archive.ics.uci.edu/ml/machine-learning-databases/balloons/balloons.names"}
standard_datasets["bank_marketing"] = {"data": "http://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip",
                                       "meta": "http://archive.ics.uci.edu/ml/datasets/Bank+Marketing"}
standard_datasets["banknote_authentication"] = {"data": "http://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt",
                                                "meta": "http://archive.ics.uci.edu/ml/datasets/banknote+authentication"}
standard_datasets["car_evaluation"] = {"data": ["http://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data",
                                                "http://archive.ics.uci.edu/ml/machine-learning-databases/car/car.c45-names"],
                                       "meta": "http://archive.ics.uci.edu/ml/machine-learning-databases/car/car.names"}
standard_datasets["census_income_kdd"] = {"data": ["http://archive.ics.uci.edu/ml/machine-learning-databases/census-income-mld/census-income.data.gz",
                                                   "http://archive.ics.uci.edu/ml/machine-learning-databases/census-income-mld/census-income.test.gz",
                                                   "http://archive.ics.uci.edu/ml/machine-learning-databases/census-income-mld/census.tar.gz"],
                                          "meta": "http://archive.ics.uci.edu/ml/machine-learning-databases/census-income-mld/census-income.data.html"}
standard_datasets["chess_king_rook_vs_king"] = {"data": "http://archive.ics.uci.edu/ml/machine-learning-databases/chess/king-rook-vs-king/krkopt.data",
                                                "meta": "http://archive.ics.uci.edu/ml/machine-learning-databases/chess/king-rook-vs-king/krkopt.info"}
standard_datasets["climate_model_crashes"] = {"data": "http://archive.ics.uci.edu/ml/machine-learning-databases/00252/pop_failures.dat",
                                              "meta": "http://archive.ics.uci.edu/ml/datasets/Climate+Model+Simulation+Crashes"}
standard_datasets["connect4"] = {"data": "http://archive.ics.uci.edu/ml/machine-learning-databases/connect-4/connect-4.data.Z",
                                 "meta": "http://archive.ics.uci.edu/ml/machine-learning-databases/connect-4/connect-4.names"}
standard_datasets["covertype"] = {"data": "http://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz",
                                  "meta": "http://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.info"}


"""
Datasets not used
  :AutoUniv Data Set: http://archive.ics.uci.edu/ml/datasets/AutoUniv
    This is just a dataset generator
  :Blogger: http://archive.ics.uci.edu/ml/datasets/BLOGGER
    Dataset is provided as an Excel sheet
  :Breast Tissue: http://archive.ics.uci.edu/ml/datasets/Breast+Tissue
    Dataset is provided as an Excel sheet
  :Cardiotography: http://archive.ics.uci.edu/ml/datasets/Cardiotocography
    Dataset is provided as an Excel sheet
  :Census Income Data Set: http://archive.ics.uci.edu/ml/datasets/Census+Income
    Same as adult data set
  :Character Trajectories Data Set: http://archive.ics.uci.edu/ml/datasets/Character+Trajectories
    Goal is to classifiy a whole pentip time-series
  :Chess (King-Rook vs. King-Knight): http://archive.ics.uci.edu/ml/datasets/Chess+%28King-Rook+vs.+King-Knight%29
    A generator
  :Connectionist Bench: http://archive.ics.uci.edu/ml/datasets/Connectionist+Bench+%28Sonar%2C+Mines+vs.+Rocks%29
    Strange data format
"""