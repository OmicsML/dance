import os.path as osp

IMPUTATION_DATASET_TO_FILE = {
    "pbmc_data":
    "5k_pbmc_protein_v3_filtered_feature_bc_matrix.h5",
    "mouse_embryo_data": [
        osp.join("GSE65525", i) for i in [
            "GSM1599494_ES_d0_main.csv",
            "GSM1599497_ES_d2_LIFminus.csv",
            "GSM1599498_ES_d4_LIFminus.csv",
            "GSM1599499_ES_d7_LIFminus.csv",
        ]
    ],
    "mouse_brain_data":
    "neuron_10k_v3_filtered_feature_bc_matrix.h5",
    "human_stemcell_data":
    "GSE75748/GSE75748_sc_time_course_ec.csv.gz",
    "human_breast_TGFb_data":
    "GSE114397_HMLE_TGFb.csv",
    "human_breast_Dox_data":
    "GSM3141014_Zeb1_Dox.csv",
    "human_melanoma_data":
    "human_melanoma_data.csv",
    "mouse_visual_data": [
        'GSM2746905_B4_11_0h_counts.csv',
        # 'GSM2746906_B4_12_0h_counts.csv',
        # 'GSM2746922_B7_23_4h_B_counts.csv',
        # 'GSM2746895_B1_1_0h_counts.csv',
        # 'GSM2746916_B6_20_4h_A_counts.csv',
        # 'GSM2746903_B3_9_4h_counts.csv',
        # 'GSM2746914_B6_19_4h_A_counts.csv',
        # 'GSM2746908_B5_14_0h_counts.csv',
        # 'GSM2746907_B5_13_0h_counts.csv',
        # 'GSM2746917_B6_20_4h_B_counts.csv',
        # 'GSM2746918_B7_21_1h_counts.csv',
        # 'GSM2746898_B2_4_1h_counts.csv',
        # 'GSM2746909_B5_15_0h_counts.csv',
        # 'GSM2746915_B6_19_4h_B_counts.csv',
        # 'GSM2746897_B1_3_4h_counts.csv',
        # 'GSM2746902_B3_8_1h_counts.csv',
        # 'GSM2746911_B6_17_1h_A_counts.csv',
        # 'GSM2746904_B3_10_4h_counts.csv',
        # 'GSM2746900_B3_6_0h_counts.csv',
        # 'GSM2746920_B7_22_4h_B_counts.csv',
        # 'GSM2746896_B1_2_1h_counts.csv',
        # 'GSM2746921_B7_23_4h_A_counts.csv',
        # 'GSM2746899_B3_5_0h_counts.csv',
        # 'GSM2746919_B7_22_4h_A_counts.csv',
        # 'GSM2746901_B3_7_1h_counts.csv',
        # 'GSM2746910_B5_16_0h_counts.csv',
        # 'GSM2746912_B6_17_1h_B_counts.csv',
        'GSM2746913_B6_18_1h_counts.csv'
    ]
}
