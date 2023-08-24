import collections
import os
import os.path as osp
import pprint
import shutil
import sys
from glob import glob

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import csr_matrix

from dance import logger
from dance.data import Data
from dance.datasets.base import BaseDataset
from dance.registers import register_dataset
from dance.typing import Dict, List, Optional, Set, Tuple
from dance.utils.download import download_file, download_unzip
from dance.utils.preprocess import cell_label_to_df


@register_dataset("scdeepsort")
class ScDeepSortDataset(BaseDataset):

    _DISPLAY_ATTRS = ("species", "tissue", "train_dataset", "test_dataset")
    ALL_URL_DICT: Dict[str, str] = {
        "train_human_cell_atlas":   "https://www.dropbox.com/s/1itq1pokplbqxhx?dl=1",
        "test_human_test_data":     "https://www.dropbox.com/s/gpxjnnvwyblv3xb?dl=1",
        "train_mouse_cell_atlas":   "https://www.dropbox.com/s/ng8d3eujfah9ppl?dl=1",
        "test_mouse_test_data":     "https://www.dropbox.com/s/pkr28czk5g3al2p?dl=1",
    }  # yapf: disable
    BENCH_URL_DICT: Dict[str, str] = {
        # Mouse spleen benchmark
        "train_mouse_Spleen1970_celltype.csv":  "https://www.dropbox.com/s/3ea64vk546fjxvr?dl=1",
        "train_mouse_Spleen1970_data.csv":      "https://www.dropbox.com/s/c4te0fr1qicqki8?dl=1",
        "test_mouse_Spleen1759_celltype.csv":   "https://www.dropbox.com/s/gczehvgai873mhb?dl=1",
        "test_mouse_Spleen1759_data.csv":       "https://www.dropbox.com/s/fl8t7rbo5dmznvq?dl=1",

        # Mouse brain benchmark
        "train_mouse_Brain753_celltype.csv":    "https://www.dropbox.com/s/x2katwk93z06sgw?dl=1",
        "train_mouse_Brain753_data.csv":        "https://www.dropbox.com/s/3f3wbplgo3xa4ww?dl=1",
        "train_mouse_Brain3285_celltype.csv":   "https://www.dropbox.com/s/ozsobozk3ihkrqg?dl=1",
        "train_mouse_Brain3285_data.csv":       "https://www.dropbox.com/s/zjrloejx8iqdqsa?dl=1",
        "test_mouse_Brain2695_celltype.csv":    "https://www.dropbox.com/s/gh72dk7i0p7fggu?dl=1",
        "test_mouse_Brain2695_data.csv":        "https://www.dropbox.com/s/ufianih66xjqxdu?dl=1",
        # Mouse kidney benchmark
        "train_mouse_Kidney4682_celltype.csv":  "https://www.dropbox.com/s/3plrve7g9v428ec?dl=1",
        "train_mouse_Kidney4682_data.csv":      "https://www.dropbox.com/s/olf5nirtieu1ikq?dl=1",
        "test_mouse_Kidney203_celltype.csv":    "https://www.dropbox.com/s/t4eyaig889qdiz2?dl=1",
        "test_mouse_Kidney203_data.csv":        "https://www.dropbox.com/s/kmos1ceubumgmpj?dl=1",
        "train_human_Brain328_celltype.csv":"https://dl.dropboxusercontent.com/scl/fi/6s5c2tw1pm4cqadyiysnh/human_Brain328_celltype.csv?rlkey=8b8a1jvmyf8zr52rogiii1zxj&dl=0",
        "train_human_Brain328_data.csv":"https://dl.dropboxusercontent.com/scl/fi/qy09gjzapmsrqnk6v8g8e/human_Brain328_data.csv?rlkey=ykt4dvbyz12ei0gcld4rtrcfb&dl=0",
        "test_human_Brain138_celltype.csv":"https://dl.dropboxusercontent.com/scl/fi/umh8n798z3xkjiatmbb5n/human_Brain138_celltype.csv?rlkey=30mafnfp2o9e1xyv2rhy8vxj6&dl=0",
        "test_human_Brain138_data.csv":"https://dl.dropboxusercontent.com/scl/fi/ayhbmxrqiqjbw39u6n54o/human_Brain138_data.csv?rlkey=o2vse9qbmjl3bc4van8ru26ga&dl=0",
        "train_human_Spleen3043_celltype.csv": "https://dl.dropboxusercontent.com/scl/fi/18w6ikivkiqvot88kost6/human_Spleen3043_celltype.csv?rlkey=14k438vfhfsoijla8ngdmmvck&dl=0",
        "train_human_Spleen3043_data.csv": "https://dl.dropboxusercontent.com/scl/fi/6630o9i9ln9i9nq013gz1/human_Spleen3043_data.csv?rlkey=qme5ho206nl9aaruiwsbpkxcj&dl=0",
        "train_human_Spleen4657_celltype.csv":"https://dl.dropboxusercontent.com/scl/fi/udfwibe7itemp5e0lhcxc/human_Spleen4657_celltype.csv?rlkey=n65vmrov4gbhltzc1itc6n72d&dl=0",
        "train_human_Spleen4657_data.csv":"https://dl.dropboxusercontent.com/scl/fi/je635zn3zlj7xe3n0pgr8/human_Spleen4657_data.csv?rlkey=umkslvwow9asjn2n1udu91y0a&dl=0",
        "train_human_Spleen4362_celltype.csv":"https://dl.dropboxusercontent.com/scl/fi/qdhdhtcpsuxq24kswinlc/human_Spleen4362_celltype.csv?rlkey=9gxd5gjumax909i6yxlb8mh5i&dl=0",
        "train_human_Spleen4362_data.csv":"https://dl.dropboxusercontent.com/scl/fi/nj48cfxsxajortt5hvb5j/human_Spleen4362_data.csv?rlkey=d0k9vdvvjd38yzc7bwojiddsn&dl=0",
        "train_human_Spleen4115_celltype.csv":"https://dl.dropboxusercontent.com/scl/fi/5puarvjrru4bziscpsdrf/human_Spleen4115_celltype.csv?rlkey=dwdb42zah5qj7bnx2p1373hf7&dl=0",
        "train_human_Spleen4115_data.csv":"https://dl.dropboxusercontent.com/scl/fi/0zfkl5fhyz5fac4xrnwk6/human_Spleen4115_data.csv?rlkey=6flhn560buz31ay53vo0w1cvh&dl=0",
        "train_human_Spleen4029_celltype.csv":"https://dl.dropboxusercontent.com/scl/fi/zl0remocq8zxs6t0cjryo/human_Spleen4029_celltype.csv?rlkey=tkkg0sa1pe1e5i9dwt5twnuov&dl=0",
        "train_human_Spleen4029_data.csv":"https://dl.dropboxusercontent.com/scl/fi/x1es7uiexm1bhf0a4dt6t/human_Spleen4029_data.csv?rlkey=fxfxupkxrnhza5cbfclyd7nnm&dl=0",
        "train_human_Spleen3777_celltype.csv":"https://dl.dropboxusercontent.com/scl/fi/en0t9agskuclxneamk59p/human_Spleen3777_celltype.csv?rlkey=hzkd9r34o4hqo7mayb78bs5ut&dl=0",
        "train_human_Spleen3777_data.csv":"https://dl.dropboxusercontent.com/scl/fi/7l55kajfgo1da3ftv4q28/human_Spleen3777_data.csv?rlkey=8lboh1qp21impmlbls07s1wr8&dl=0",
        "test_human_Spleen1729_celltype.csv": "https://dl.dropboxusercontent.com/scl/fi/i8zo1l7mpah9r11zh97bb/human_Spleen1729_celltype.csv?rlkey=aibwl441j460x6l8jxyvfeedx&dl=0",
        "test_human_Spleen1729_data.csv": "https://dl.dropboxusercontent.com/scl/fi/28n7eqabdss7c58nfqz67/human_Spleen1729_data.csv?rlkey=rdjgx5ndmfiqat796k3t1zi3k&dl=0",
        "test_human_Spleen2125_celltype.csv":"https://dl.dropboxusercontent.com/scl/fi/9qd1pdy1lqt5e3zvr748w/human_Spleen2125_celltype.csv?rlkey=hkotuf1d4h1yx4w1kiu12q4ei&dl=0",
        "test_human_Spleen2125_data.csv":"https://dl.dropboxusercontent.com/scl/fi/y5hzova8979nnwpdrd9s4/human_Spleen2125_data.csv?rlkey=dn1j8ydhahj5v9yf8do1jav98&dl=0",
        "test_human_Spleen2184_celltype.csv":"https://dl.dropboxusercontent.com/scl/fi/x5rmz0llbll1bvwk8hh50/human_Spleen2184_celltype.csv?rlkey=i28lfi1ogig7wwzogm26lvq00&dl=0",
        "test_human_Spleen2184_data.csv":"https://dl.dropboxusercontent.com/scl/fi/vpod0zvpn2tnvm1if8pnh/human_Spleen2184_data.csv?rlkey=1lz6d5n82g7cg0ynlw2mwrvik&dl=0",
        "test_human_Spleen2724_celltype.csv": "https://dl.dropboxusercontent.com/scl/fi/3kor4ezq4b8kn0bzf5yh1/human_Spleen2724_celltype.csv?rlkey=4jzinot3a95o7a3jv1jwvb2zm&dl=0",
        "test_human_Spleen2724_data.csv": "https://dl.dropboxusercontent.com/scl/fi/ns8n3amo4oipw2lhd90md/human_Spleen2743_data.csv?rlkey=5tax6jxge37x14elhwv8b1bi5&dl=0",
        "test_human_Spleen2743_celltype.csv": "https://dl.dropboxusercontent.com/scl/fi/eo86chn0acfuzvj947i8j/human_Spleen2743_celltype.csv?rlkey=y3mmkp9v05ei1hvpw1tvemffc&dl=0",
        "test_human_Spleen2743_data.csv": "https://dl.dropboxusercontent.com/scl/fi/ns8n3amo4oipw2lhd90md/human_Spleen2743_data.csv?rlkey=5tax6jxge37x14elhwv8b1bi5&dl=0",
        "train_human_Immune11407_celltype.csv":"https://dl.dropboxusercontent.com/scl/fi/mcw025mbv82fz19ihqabp/human_Immune11407_celltype.csv?rlkey=kx03krmbz1i91gbsl6pz4m9tk&dl=0",
        "train_human_Immune11407_data.csv": "https://dl.dropboxusercontent.com/scl/fi/t5qy70vjvp9ag3jfnbeh3/human_Immune11407_data.csv?rlkey=zq0l1e83jh2wcumzrw40d0w0i&dl=0",
        "train_human_Immune9258_celltype.csv":"https://dl.dropboxusercontent.com/scl/fi/fse8yheqap91ksru4v26q/human_Immune9258_celltype.csv?rlkey=3ky50q0770kpifc8z1i85lgus&dl=0",
        "train_human_Immune9258_data.csv":"https://dl.dropboxusercontent.com/scl/fi/g9k6hd1suooy2p6os49f2/human_Immune9258_data.csv?rlkey=87tn0g5rzmpjp3r6hkat86fe5&dl=0",
        "train_human_Immune9054_celltype.csv": "https://dl.dropboxusercontent.com/scl/fi/ei0jlbuekmkuyl47crf9u/human_Immune9054_celltype.csv?rlkey=xa162r5s8wjin5nx7s0ua0n15&dl=0",
        "train_human_Immune9054_data.csv": "https://dl.dropboxusercontent.com/scl/fi/cfoop5xtdy4heoucyoewk/human_Immune9054_data.csv?rlkey=dm3c9b5q686d2hwxupdil5w04&dl=0",
        "train_human_Immune1519_celltype.csv": "https://dl.dropboxusercontent.com/scl/fi/ct6lofsy2iiincbxesgva/human_Immune1519_celltype.csv?rlkey=hkmd9wicuhqb77gfsmhsvrwqe&dl=0",
        "train_human_Immune1519_data.csv": "https://dl.dropboxusercontent.com/scl/fi/sv3su043zvpyylt7vxrnl/human_Immune1519_data.csv?rlkey=0v4kak6z00ms1fgzg3wcjjgvg&dl=0",
        "train_human_Immune713_celltype.csv": "https://dl.dropboxusercontent.com/scl/fi/yb8eup040p6gsx5pkldrx/human_Immune713_celltype.csv?rlkey=zijh9zo8pl8nt1lllpeg7d0wq&dl=0",
        "train_human_Immune713_data.csv": "https://dl.dropboxusercontent.com/scl/fi/zococ4r9fy75iycdxl5zi/human_Immune713_data.csv?rlkey=h66j5px6y30syws2flhz04z1v&dl=0",
        "train_human_Immune636_celltype.csv": "https://dl.dropboxusercontent.com/scl/fi/zmger4aw2aj5xj2gxxh6x/human_Immune636_celltype.csv?rlkey=ryjkwoxsdjp8wzu6oy2e0n3y0&dl=0",
        "train_human_Immune636_data.csv": "https://dl.dropboxusercontent.com/scl/fi/7nsrxxtj1qorh8644tpgc/human_Immune636_data.csv?rlkey=oszaqv9oqnv0vyi2tm0a0id75&dl=0",
        "test_human_Immune7572_celltype.csv":"https://dl.dropboxusercontent.com/scl/fi/p9xi5xwql2sir5x1uilkh/human_Immune7572_celltype.csv?rlkey=qwnt7hf7j9foun8hrp7yiccug&dl=0",
        "test_human_Immune7572_data.csv":"https://dl.dropboxusercontent.com/scl/fi/kc8feu239we1g44c2vi15/human_Immune7572_data.csv?rlkey=e2bpcog294u5niiralbtuuwps&dl=0",
        "test_human_Immune6509_celltype.csv":"https://dl.dropboxusercontent.com/scl/fi/cst6c83zlgo3jizy71kid/human_Immune6509_celltype.csv?rlkey=m335tj8zoi2kjx2h9n0k027cp&dl=0",
        "test_human_Immune6509_data.csv":"https://dl.dropboxusercontent.com/scl/fi/hms2pd90x4ooat0878vah/human_Immune6509_data.csv?rlkey=kg8hcx6xpbsyo9xyc4wv79wtq&dl=0",
        "test_human_Immune3323_celltype.csv":"https://dl.dropboxusercontent.com/scl/fi/z9otfdaz8amat8ni5p2ek/human_Immune3323_celltype.csv?rlkey=klnb9r5huizi6eyqkhkbqihcl&dl=0",
        "test_human_Immune3323_data.csv":"https://dl.dropboxusercontent.com/scl/fi/voyzalx45u5kp6f5z3fsq/human_Immune3323_data.csv?rlkey=j4b3ygwgy985domu9nkxc9nko&dl=0",
        "test_human_Immune1925_celltype.csv": "https://dl.dropboxusercontent.com/scl/fi/qweb0156wcfsb2gwqlwtd/human_Immune1925_celltype.csv?rlkey=dlbmz63ph0ss7l6d3b0a7wipm&dl=0",
        "test_human_Immune1925_data.csv": "https://dl.dropboxusercontent.com/scl/fi/nn1tazw4gj3aabdu1nm4o/human_Immune1925_data.csv?rlkey=9kh2yrbdzj2l1tdj20tbxbyfb&dl=0",
        "test_human_Immune205_celltype.csv": "https://dl.dropboxusercontent.com/scl/fi/n6aa3rn4is2cjqy5bx51y/human_Immune205_celltype.csv?rlkey=inojmgblnjkn8ya2cfynhyfiz&dl=0",
        "test_human_Immune205_data.csv": "https://dl.dropboxusercontent.com/scl/fi/veawggfxoqk2xp98av1r0/human_Immune205_data.csv?rlkey=85hc4bm4h3ynt3khce3ncj8t9&dl=0",
        "train_human_CD41247_celltype.csv":"https://dl.dropboxusercontent.com/scl/fi/6pccnff27e8bpzkpbs30d/human_CD41247_celltype.csv?rlkey=ll6uipqapx38gfrmcyd4dzy7j&dl=0",
        "train_human_CD41247_data.csv":"https://dl.dropboxusercontent.com/scl/fi/a2cw89rksiyymm85s578o/human_CD41247_data.csv?rlkey=rwx79a4nra4t1d672dkijuf2k&dl=0",
        "train_human_CD41013_celltype.csv":"https://dl.dropboxusercontent.com/scl/fi/w8rtm5lskqz6t0zdeuhdd/human_CD41013_celltype.csv?rlkey=t2a7uz7j9d3uvt5qm43ju7m3g&dl=0",
        "train_human_CD41013_data.csv":"https://dl.dropboxusercontent.com/scl/fi/jlvdmz9fark313bxjxn93/human_CD41013_data.csv?rlkey=dv7xl7d1yqkozdu2sx20gravg&dl=0",
        "train_human_CD4864_celltype.csv": "https://dl.dropboxusercontent.com/scl/fi/s8tj6841njnyhymrnun5i/human_CD4864_celltype.csv?rlkey=ofuke8w7f7oo7mjlo1qhxjfkz&dl=0",
        "train_human_CD4864_data.csv": "https://dl.dropboxusercontent.com/scl/fi/em4rmlonlhyzoohko7z69/human_CD4864_data.csv?rlkey=x9djiz0yd8wmrt9swlkijrbg0&dl=0",
        "train_human_CD4845_celltype.csv": "https://dl.dropboxusercontent.com/scl/fi/i2xc4qckt83lkobi3rgqk/human_CD4845_celltype.csv?rlkey=l8oixna8iokpupjrqseedef8k&dl=0",
        "train_human_CD4845_data.csv": "https://dl.dropboxusercontent.com/scl/fi/mhhpavl45jqxmc9ut9pgr/human_CD4845_data.csv?rlkey=kqxtudhlq7opx4e9dk4duttut&dl=0",
        "train_human_CD4784_celltype.csv": "https://dl.dropboxusercontent.com/scl/fi/rtzue5poq59cgbni7d50z/human_CD4784_celltype.csv?rlkey=lvgliz8po261h8hvyg07me0po&dl=0",
        "train_human_CD4784_data.csv": "https://dl.dropboxusercontent.com/scl/fi/09ks8l2b7kjt7xdbsdm73/human_CD4784_data.csv?rlkey=70lkdzd4xmnwzz9d7908m0pi2&dl=0",
        "train_human_CD4770_celltype.csv": "https://dl.dropboxusercontent.com/scl/fi/ndew7n2ph5ywyl43n2no1/human_CD4770_celltype.csv?rlkey=gl6p2cug6rzuzsnbk1w0y9fjr&dl=0",
        "train_human_CD4770_data.csv": "https://dl.dropboxusercontent.com/scl/fi/5bcgribz8ygii44ntbptx/human_CD4770_data.csv?rlkey=st3o4xkbk4sas9nty3ibx4pi4&dl=0",
        "train_human_CD4768_celltype.csv": "https://dl.dropboxusercontent.com/scl/fi/lk1vss7l792nmuc5nespu/human_CD4768_celltype.csv?rlkey=2i72zu7eijopknwppso96jf8n&dl=0",
        "train_human_CD4768_data.csv": "https://dl.dropboxusercontent.com/scl/fi/xxzxpkovbxookki4aowni/human_CD4768_data.csv?rlkey=ah0czg3dxxazu38ahe92m38jk&dl=0",
        "train_human_CD4767_celltype.csv": "https://dl.dropboxusercontent.com/scl/fi/1bwpkn7dvihl70kmqikqs/human_CD4767_celltype.csv?rlkey=9nrv18yyfvmwnf3xwprmh1362&dl=0",
        "train_human_CD4767_data.csv": "https://dl.dropboxusercontent.com/scl/fi/4eur0f00vqxso81crym7t/human_CD4767_data.csv?rlkey=vdrv4yy6zx3vsg4zlqf7kf0xp&dl=0",
        "train_human_CD4732_celltype.csv": "https://dl.dropboxusercontent.com/scl/fi/eh1vzdfknn0635b9laupd/human_CD4732_celltype.csv?rlkey=mh4x4nw3tuev31pjt4eqrnklt&dl=0",
        "train_human_CD4732_data.csv": "https://dl.dropboxusercontent.com/scl/fi/hacspep5ahvpiuybhu7di/human_CD4732_data.csv?rlkey=gqiu41oot842w5ie536x8sqb2&dl=0",
        "train_human_CD4598_celltype.csv": "https://dl.dropboxusercontent.com/scl/fi/ql9ley85gw8evkaapr614/human_CD4598_celltype.csv?rlkey=mpbquj4lo775e37rnq0zjjw2m&dl=0",
        "train_human_CD4598_data.csv": "https://dl.dropboxusercontent.com/scl/fi/svbajge0fo1jky37vphcc/human_CD4598_data.csv?rlkey=6hr43oyt0l2gi92tjals8bbov&dl=0",
        "test_human_CD4559_celltype.csv":"https://dl.dropboxusercontent.com/scl/fi/1py17zrzojl7wbmp11mg4/human_CD4559_celltype.csv?rlkey=kjo5535a36e6515jcyn35ai1t&dl=0",
        "test_human_CD4559_data.csv":"https://dl.dropboxusercontent.com/scl/fi/tu1h3qxotyype5rcrz90o/human_CD4559_data.csv?rlkey=as28pxvstosjrejt083cvxsxt&dl=0",
        "test_human_CD4551_celltype.csv":"https://dl.dropboxusercontent.com/scl/fi/1vha3g6elts40sfrlrn37/human_CD4551_celltype.csv?rlkey=32ttum88nolf25i6igirrss5m&dl=0",
        "test_human_CD4551_data.csv": "https://dl.dropboxusercontent.com/scl/fi/3b5bcg8ldbrymk6zth65c/human_CD4551_data.csv?rlkey=cqw24clr8l64fb3oawavp6l1a&dl=0",
        "test_human_CD4490_celltype.csv": "https://dl.dropboxusercontent.com/scl/fi/e3y40spw8x0bcwaxl5ord/human_CD4490_celltype.csv?rlkey=219pee8krxcrxdgg6ww7tqwau&dl=0",
        "test_human_CD4490_data.csv": "https://dl.dropboxusercontent.com/scl/fi/66nwl59wxtuithbkrjarm/human_CD4490_data.csv?rlkey=tbuj2qhzilfh9n5a18qrkhu04&dl=0",
        "test_human_CD4437_celltype.csv": "https://dl.dropboxusercontent.com/scl/fi/bsz7no10u5gj1r92209gt/human_CD4437_celltype.csv?rlkey=wz4dqijn1jaaamph1xs41u3dc&dl=0",
        "test_human_CD4437_data.csv": "https://dl.dropboxusercontent.com/scl/fi/b7dpywbb7b7ena8d64ncc/human_CD4437_data.csv?rlkey=b8afnfnttg4my0u6ll4vayfoj&dl=0",
        "test_human_CD4404_celltype.csv": "https://dl.dropboxusercontent.com/scl/fi/nbbc51i4zsh5qzgvgkbrq/human_CD4404_celltype.csv?rlkey=cf6b89p7i41fdi477ddpda4kz&dl=0",
        "test_human_CD4404_data.csv": "https://dl.dropboxusercontent.com/scl/fi/rfj5kdvosgwwkm1dsxfd0/human_CD4404_data.csv?rlkey=wyaz5zizwe9dkpsivzuw8gyfz&dl=0",
        "test_human_CD4390_celltype.csv": "https://dl.dropboxusercontent.com/scl/fi/tfycm4df9m3f0xqywvn8l/human_CD4390_celltype.csv?rlkey=0ho1q8ia7r18hi0vzcl060g4a&dl=0",
        "test_human_CD4390_data.csv": "https://dl.dropboxusercontent.com/scl/fi/yzvb7thg8nizbou6yanuo/human_CD4390_data.csv?rlkey=fmhgez0l91je9rucs2kg1cmrm&dl=0",
        "test_human_CD4381_celltype.csv": "https://dl.dropboxusercontent.com/scl/fi/xid3tohlf5kbz8dpvzg5y/human_CD4381_celltype.csv?rlkey=qgvcpitjcrzrnysquaw9ofw0y&dl=0",
        "test_human_CD4381_data.csv": "https://dl.dropboxusercontent.com/scl/fi/aoz03m2lhqfh6md0034o2/human_CD4381_data.csv?rlkey=gkws8ntvrpmc34codedesnuhx&dl=0",
        "test_human_CD4376_celltype.csv": "https://dl.dropboxusercontent.com/scl/fi/0br0bj7vx6s8gznz8khav/human_CD4376_celltype.csv?rlkey=qndtz6j7by46nl7ilhuedbt9s&dl=0",
        "test_human_CD4376_data.csv": "https://dl.dropboxusercontent.com/scl/fi/iaw2zcz1f5vffytqajw88/human_CD4376_data.csv?rlkey=65kn3z7wijsvflclip8lg7qrw&dl=0",
        "test_human_CD4340_celltype.csv": "https://dl.dropboxusercontent.com/scl/fi/568d6pyzmnbw8q1g24czw/human_CD4340_celltype.csv?rlkey=s6za3tkohu1hhiz9485vkpalp&dl=0",
        "test_human_CD4340_data.csv": "https://dl.dropboxusercontent.com/scl/fi/jnfen8pedmhqj6c9spgju/human_CD4340_data.csv?rlkey=fqeg1wjmetfa48zazaux3v5z1&dl=0",
        "test_human_CD4315_celltype.csv": "https://dl.dropboxusercontent.com/scl/fi/uznry9km4q32vzm0fs9zh/human_CD4315_celltype.csv?rlkey=yd3q8j0fnp4zdtjifcg35qgoi&dl=0",
        "test_human_CD4315_data.csv": "https://dl.dropboxusercontent.com/scl/fi/r8fj7t5xsng68wlodfjiw/human_CD4315_data.csv?rlkey=nv55kwsirmsjh0hvmiy1kpo9n&dl=0",
        "train_human_CD81641_celltype.csv":"https://dl.dropboxusercontent.com/scl/fi/9wt6lvubklrca4mpur9rl/human_CD81641_celltype.csv?rlkey=30fn1vmfp9lbcxl4ughkqvjl5&dl=0",
        "train_human_CD81641_data.csv":"https://dl.dropboxusercontent.com/scl/fi/6ya70xs0s4aqzkjnkrs9s/human_CD81641_data.csv?rlkey=qku4kqu67b55xqpx1cchx4apx&dl=0",
        "train_human_CD81357_celltype.csv":"https://dl.dropboxusercontent.com/scl/fi/cufeihloz1slnqj2cj36k/human_CD81357_celltype.csv?rlkey=s9whz8wlm1q2aunrsx72ee1dl&dl=0",
        "train_human_CD81357_data.csv":"https://dl.dropboxusercontent.com/Sscl/fi/010rmb16aoeno4km67n41/human_CD81357_data.csv?rlkey=g7qm8v6shc6k0vgiez4envumw&dl=0",
        "train_human_CD81027_celltype.csv":"https://dl.dropboxusercontent/scl/fi/9r6tvbyb1p9al7k4i912y/human_CD81027_celltype.csv?rlkey=y7i7urv4mvksb382gm85pm75o&dl=0",
        "train_human_CD81027_data.csv": "https://dl.dropboxusercontent/scl/fi/f2jldx2pc8gnynukrb21k/human_CD81027_data.csv?rlkey=grmh4kcno0onzdb9rzqdwa2pp&dl=0",
        "train_human_CD8972_celltype.csv": "https://dl.dropboxusercontent/scl/fi/8k8njt5xvo39qcvo8fx67/human_CD8972_celltype.csv?rlkey=v5tphei7mk9upjqudlde3039f&dl=0",
        "train_human_CD8972_data.csv": "https://dl.dropboxusercontent/scl/fi/6h2cefm8diinsem2katxd/human_CD8972_data.csv?rlkey=icpu4eu4hgcv8i46keo3pw07w&dl=0",
        "train_human_CD8850_celltype.csv": "https://dl.dropboxusercontent/scl/fi/ian4tfu7gqdsil4801h9g/human_CD8850_celltype.csv?rlkey=eb7hx9xzdvinbgn66ozzidcu5&dl=0",
        "train_human_CD8850_data.csv": "https://dl.dropboxusercontent/scl/fi/bhog439jdasqw8949pyig/human_CD8850_data.csv?rlkey=yljgy72v1j2we7j291u0a5m2a&dl=0",
        "train_human_CD8777_celltype.csv": "https://dl.dropboxusercontent/scl/fi/j2g2b3o4695c6fzvlqnme/human_CD8777_celltype.csv?rlkey=1jjuz1x19e9b08rfso88noce0&dl=0",
        "train_human_CD8777_data.csv": "https://dl.dropboxusercontent/scl/fi/m005ke1c8ceslnvyew1db/human_CD8777_data.csv?rlkey=cvevpxqvazlvbiw36cm0xfrfk&dl=0",
        "train_human_CD8706_celltype.csv": "https://dl.dropboxusercontent/scl/fi/kuscal3a1lfd0ybg0ffxi/human_CD8706_celltype.csv?rlkey=0akpocxdxgjvt59k49aefb31u&dl=0",
        "train_human_CD8706_data.csv": "https://dl.dropboxusercontent/scl/fi/cbvq03rm9jdwyermjb2y5/human_CD8706_data.csv?rlkey=0666w8ld4rmhkxrpyj9rqbk0n&dl=0",
        "train_human_CD8517_celltype.csv": "https://dl.dropboxusercontent/scl/fi/o5nfy8s98hzxn6mgh2omn/human_CD8517_celltype.csv?rlkey=wgqtwa2awdo4s7ahb3hwfcqhd&dl=0",
        "train_human_CD8517_data.csv": "https://dl.dropboxusercontent/scl/fi/dxmaww8uwoitvodylirls/human_CD8517_data.csv?rlkey=9lru1dnd1oy3g0ir7dsx0et5d&dl=0",
        "test_human_CD8492_celltype.csv":"https://dl.dropboxusercontent.com/scl/fi/0j5d33mnwpg31ldwq6c3k/human_CD8492_celltype.csv?rlkey=58rlv21lc63dzjs1upxzzb7z1&dl=0",
        "test_human_CD8492_data.csv":"https://dl.dropboxusercontent.com/scl/fi/flbgu5x5us4aqdai6dm4g/human_CD8492_data.csv?rlkey=ytawomsnumf62zjvq70i5p8hm&dl=0",
        "test_human_CD8470_celltype.csv":"https://dl.dropboxusercontent.com/scl/fi/dg4jcxvoq6hsw4ec8epvr/human_CD8470_celltype.csv?rlkey=irzpdywjg9wq5504cok3vt940&dl=0",
        "test_human_CD8470_data.csv": "https://dl.dropboxusercontent.com/scl/fi/euluacs9igllqkav1nyqp/human_CD8470_data.csv?rlkey=qojh0b1mdzb4f54voyqfuozna&dl=0",
        "test_human_CD8455_celltype.csv": "https://dl.dropboxusercontent.com/scl/fi/iprfo1r2r0ii850v4qgkm/human_CD8455_celltype.csv?rlkey=nytwswyr4f8zisc60mh2bzgdt&dl=0",
        "test_human_CD8455_data.csv": "https://dl.dropboxusercontent.com/scl/fi/unh47vcwccc9cwre4j5xc/human_CD8455_data.csv?rlkey=ewh27p23jc7iw6wdmqk8tkl26&dl=0",
        "test_human_CD8405_celltype.csv": "https://dl.dropboxusercontent.com/scl/fi/3m38ky0w7romd9m5xh1dg/human_CD8405_celltype.csv?rlkey=tebjsx8yqv8i9gpxe7tygk1w4&dl=0",
        "test_human_CD8405_data.csv": "https://dl.dropboxusercontent.com/scl/fi/y1lxlw5436e1qzfy1tmdf/human_CD8405_data.csv?rlkey=g4zu9cakd7rxg6kah7mshfo12&dl=0",
        "test_human_CD8398_celltype.csv": "https://dl.dropboxusercontent.com/scl/fi/4l7crrgee89sgzxl36z8r/human_CD8398_celltype.csv?rlkey=bfackypwryxq8ljdwwa8g8q6h&dl=0",
        "test_human_CD8398_data.csv": "https://dl.dropboxusercontent.com/scl/fi/slmjr4497pekd6s22bdbp/human_CD8398_data.csv?rlkey=0d6yec2dat916iwa50wm7raia&dl=0",
        "test_human_CD8377_celltype.csv": "https://dl.dropboxusercontent.com/scl/fi/j7tbte2p47g6hih8c3uzg/human_CD8377_celltype.csv?rlkey=39jjjpgpc4ap1va25vvkpvumj&dl=0",
        "test_human_CD8377_data.csv": "https://dl.dropboxusercontent.com/scl/fi/rudn3mvac4cwttciye55b/human_CD8377_data.csv?rlkey=6vyv6sxv9gsjy7x0qjeelj3mp&dl=0",
        "test_human_CD8332_celltype.csv": "https://dl.dropboxusercontent.com/scl/fi/d95nx28hqa5cygzkp8jth/human_CD8332_celltype.csv?rlkey=jh23eg4xwdmneqfyfc07kx418&dl=0",
        "test_human_CD8332_data.csv": "https://dl.dropboxusercontent.com/scl/fi/bfnk6rmfbcejqs2shpt52/human_CD8332_data.csv?rlkey=vms0bjwze0tso70wv0e0zl2wt&dl=0",
        "test_human_CD8245_celltype.csv": "https://dl.dropboxusercontent.com/scl/fi/uab3p3a167lxs6t3jeltt/human_CD8245_celltype.csv?rlkey=0t93girzlrp7xkdyr8zyaha9q&dl=0",
        "test_human_CD8245_data.csv": "https://dl.dropboxusercontent.com/scl/fi/h7bygcaiuo09q9ap3daro/human_CD8245_data.csv?rlkey=64d6emcqrj3fx9z5d9l46vm28&dl=0"
    }  # yapf: disable
    AVAILABLE_DATA = [
        {"split": "train", "species": "mouse", "tissue": "Brain", "dataset": "3285"},
        {"split": "train", "species": "mouse", "tissue": "Brain", "dataset": "753"},
        {"split": "train", "species": "mouse", "tissue": "Kidney", "dataset": "4682"},
        {"split": "train", "species": "mouse", "tissue": "Spleen", "dataset": "1970"},
        {"split": "train", "species": "human", "tissue": "Brain", "dataset": "328"},
        {"split": "train", "species": "human", "tissue": "Immune", "dataset": "636"},
        {"split": "train", "species": "human", "tissue": "Immune", "dataset": "713"},
        {"split": "train", "species": "human", "tissue": "Immune", "dataset": "1519"},
        {"split": "train", "species": "human", "tissue": "Immune", "dataset": "9054"},
        {"split": "train", "species": "human", "tissue": "Immune", "dataset": "9258"},
        {"split": "train", "species": "human", "tissue": "Immune", "dataset": "11407"},
        {"split": "train", "species": "human", "tissue": "CD8", "dataset": "517"},
        {"split": "train", "species": "human", "tissue": "CD8", "dataset": "706"},
        {"split": "train", "species": "human", "tissue": "CD8", "dataset": "777"},
        {"split": "train", "species": "human", "tissue": "CD8", "dataset": "850"},
        {"split": "train", "species": "human", "tissue": "CD8", "dataset": "972"},
        {"split": "train", "species": "human", "tissue": "CD8", "dataset": "1027"},
        {"split": "train", "species": "human", "tissue": "CD8", "dataset": "1357"},
        {"split": "train", "species": "human", "tissue": "CD8", "dataset": "1641"},
        {"split": "train", "species": "human", "tissue": "CD4", "dataset": "598"},
        {"split": "train", "species": "human", "tissue": "CD4", "dataset": "732"},
        {"split": "train", "species": "human", "tissue": "CD4", "dataset": "767"},
        {"split": "train", "species": "human", "tissue": "CD4", "dataset": "768"},
        {"split": "train", "species": "human", "tissue": "CD4", "dataset": "770"},
        {"split": "train", "species": "human", "tissue": "CD4", "dataset": "784"},
        {"split": "train", "species": "human", "tissue": "CD4", "dataset": "845"},
        {"split": "train", "species": "human", "tissue": "CD4", "dataset": "864"},
        {"split": "train", "species": "human", "tissue": "CD4", "dataset": "1013"},
        {"split": "train", "species": "human", "tissue": "CD4", "dataset": "1247"},
        {"split": "train", "species": "human", "tissue": "Spleen", "dataset": "3043"},
        {"split": "train", "species": "human", "tissue": "Spleen", "dataset": "3777"},
        {"split": "train", "species": "human", "tissue": "Spleen", "dataset": "4029"},
        {"split": "train", "species": "human", "tissue": "Spleen", "dataset": "4115"},
        {"split": "train", "species": "human", "tissue": "Spleen", "dataset": "4362"},
        {"split": "train", "species": "human", "tissue": "Spleen", "dataset": "4657"},
        {"split": "test", "species": "mouse", "tissue": "Brain", "dataset": "2695"},
        {"split": "test", "species": "mouse", "tissue": "Kidney", "dataset": "203"},
        {"split": "test", "species": "mouse", "tissue": "Spleen", "dataset": "1759"},
        {"split": "test", "species": "human", "tissue": "Brain", "dataset": "138"},
        {"split": "test", "species": "human", "tissue": "Immune", "dataset": "205"},
        {"split": "test", "species": "human", "tissue": "Immune", "dataset": "1925"},
        {"split": "test", "species": "human", "tissue": "Immune", "dataset": "3323"},
        {"split": "test", "species": "human", "tissue": "Immune", "dataset": "6509"},
        {"split": "test", "species": "human", "tissue": "Immune", "dataset": "7572"},
        {"split": "test", "species": "human", "tissue": "CD8", "dataset": "245"},
        {"split": "test", "species": "human", "tissue": "CD8", "dataset": "332"},
        {"split": "test", "species": "human", "tissue": "CD8", "dataset": "377"},
        {"split": "test", "species": "human", "tissue": "CD8", "dataset": "398"},
        {"split": "test", "species": "human", "tissue": "CD8", "dataset": "405"},
        {"split": "test", "species": "human", "tissue": "CD8", "dataset": "455"},
        {"split": "test", "species": "human", "tissue": "CD8", "dataset": "470"},
        {"split": "test", "species": "human", "tissue": "CD8", "dataset": "492"},
        {"split": "test", "species": "human", "tissue": "CD4", "dataset": "315"},
        {"split": "test", "species": "human", "tissue": "CD4", "dataset": "340"},
        {"split": "test", "species": "human", "tissue": "CD4", "dataset": "376"},
        {"split": "test", "species": "human", "tissue": "CD4", "dataset": "381"},
        {"split": "test", "species": "human", "tissue": "CD4", "dataset": "390"},
        {"split": "test", "species": "human", "tissue": "CD4", "dataset": "404"},
        {"split": "test", "species": "human", "tissue": "CD4", "dataset": "437"},
        {"split": "test", "species": "human", "tissue": "CD4", "dataset": "490"},
        {"split": "test", "species": "human", "tissue": "CD4", "dataset": "551"},
        {"split": "test", "species": "human", "tissue": "CD4", "dataset": "559"},
        {"split": "test", "species": "human", "tissue": "Spleen", "dataset": "1729"},
        {"split": "test", "species": "human", "tissue": "Spleen", "dataset": "2125"},
        {"split": "test", "species": "human", "tissue": "Spleen", "dataset": "2184"},
        {"split": "test", "species": "human", "tissue": "Spleen", "dataset": "2724"},
        {"split": "test", "species": "human", "tissue": "Spleen", "dataset": "2743"}
    ]  # yapf: disable

    def __init__(self, full_download=False, train_dataset=None, test_dataset=None, species=None, tissue=None,
                 train_dir="train", test_dir="test", map_path="map", data_dir="./"):
        super().__init__(data_dir, full_download)

        self.data_dir = data_dir
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.species = species
        self.tissue = tissue
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.map_path = map_path

    def download_all(self):
        if self.is_complete():
            return

        # Download and overwrite
        for name, url in self.ALL_URL_DICT.items():
            download_unzip(url, self.data_dir)

            parts = name.split("_")  # [train|test]_{species}_[cell|test]_atlas
            download_path = osp.join(self.data_dir, "_".join(parts[1:]))
            move_path = osp.join(self.data_dir, *parts[:2])

            os.makedirs(osp.dirname(move_path), exist_ok=True)
            try:
                shutil.rmtree(move_path)
            except FileNotFoundError:
                pass
            os.rename(download_path, move_path)

    def download(self, download_map=True):
        if self.is_complete():
            return

        # TODO: only download missing files
        # Download training and testing data
        for name, url in self.BENCH_URL_DICT.items():
            parts = name.split("_")  # [train|test]_{species}_{tissue}{id}_[celltype|data].csv
            filename = "_".join(parts[1:])
            filepath = osp.join(self.data_dir, *parts[:2], filename)
            download_file(url, filepath)

        if download_map:
            # Download mapping data
            download_unzip("https://www.dropbox.com/sh/hw1189sgm0kfrts/AAAapYOblLApqygZ-lGo_70-a?dl=1",
                           osp.join(self.data_dir, "map"))

    def is_complete_all(self):
        """Check if data is complete."""
        check = [
            osp.join(self.data_dir, "train"),
            osp.join(self.data_dir, "test"),
            osp.join(self.data_dir, "pretrained")
        ]
        for i in check:
            if not osp.exists(i):
                logger.info(f"file {i} doesn't exist")
                return False
        return True

    def is_complete(self):
        """Check if benchmarking data is complete."""
        for name in self.BENCH_URL_DICT:
            filename = name[name.find('mouse'):]
            file_i = osp.join(self.data_dir, *name.split("_")[:2], filename)
            if not osp.exists(file_i):
                logger.info(file_i)
                logger.info(f"file {filename} doesn't exist")
                return False
        # check maps
        map_check = [
            osp.join(self.data_dir, "map", "mouse", "map.xlsx"),
            osp.join(self.data_dir, "map", "human", "map.xlsx"),
            osp.join(self.data_dir, "map", "celltype2subtype.xlsx")
        ]
        for file in map_check:
            if not osp.exists(file):
                logger.info(f"file {name} doesn't exist")
                return False
        return True

    def _load_raw_data(self, ct_col: str = "Cell_type") -> Tuple[ad.AnnData, List[Set[str]], List[str], int]:
        species = self.species
        tissue = self.tissue
        train_dataset_ids = self.train_dataset
        test_dataset_ids = self.test_dataset
        data_dir = self.data_dir
        train_dir = osp.join(data_dir, self.train_dir)
        test_dir = osp.join(data_dir, self.test_dir)
        map_path = osp.join(data_dir, self.map_path, self.species)

        # Load raw data
        train_feat_paths, train_label_paths = self._get_data_paths(train_dir, species, tissue, train_dataset_ids)
        test_feat_paths, test_label_paths = self._get_data_paths(test_dir, species, tissue, test_dataset_ids)
        train_feat, test_feat = (self._load_dfs(paths, transpose=True) for paths in (train_feat_paths, test_feat_paths))
        train_label, test_label = (self._load_dfs(paths) for paths in (train_label_paths, test_label_paths))

        # Combine features (only use features that are present in the training data)
        train_size = train_feat.shape[0]
        feat_df = pd.concat(train_feat.align(test_feat, axis=1, join="left", fill_value=0)).fillna(0)
        adata = ad.AnnData(feat_df, dtype=np.float32)

        # Convert cell type labels and map test cell type names to train
        cell_types = set(train_label[ct_col].unique())
        idx_to_label = sorted(cell_types)
        cell_type_mappings: Dict[str, Set[str]] = self.get_map_dict(map_path, tissue)
        train_labels, test_labels = train_label[ct_col].tolist(), []
        for i in test_label[ct_col]:
            test_labels.append(i if i in cell_types else cell_type_mappings.get(i))
        labels: List[Set[str]] = train_labels + test_labels

        logger.debug("Mapped test cell-types:")
        for i, j, k in zip(test_label.index, test_label[ct_col], test_labels):
            logger.debug(f"{i}:{j}\t-> {k}")

        logger.info(f"Loaded expression data: {adata}")
        logger.info(f"Number of training samples: {train_feat.shape[0]:,}")
        logger.info(f"Number of testing samples: {test_feat.shape[0]:,}")
        logger.info(f"Cell-types (n={len(idx_to_label)}):\n{pprint.pformat(idx_to_label)}")

        return adata, labels, idx_to_label, train_size

    def _raw_to_dance(self, raw_data):
        adata, cell_labels, idx_to_label, train_size = raw_data
        adata.obsm["cell_type"] = cell_label_to_df(cell_labels, idx_to_label, index=adata.obs.index)
        data = Data(adata, train_size=train_size)
        return data

    @staticmethod
    def _get_data_paths(data_dir: str, species: str, tissue: str, dataset_ids: List[str], *, filetype: str = "csv",
                        feat_suffix: str = "data", label_suffix: str = "celltype") -> Tuple[List[str], List[str]]:
        feat_paths, label_paths = [], []
        for path_list, suffix in zip((feat_paths, label_paths), (feat_suffix, label_suffix)):
            for i in dataset_ids:
                path_list.append(osp.join(data_dir, species, f"{species}_{tissue}{i}_{suffix}.{filetype}"))
        return feat_paths, label_paths

    @staticmethod
    def _load_dfs(paths: List[str], *, index_col: Optional[int] = 0, transpose: bool = False, **kwargs):
        dfs = []
        for path in paths:
            logger.info(f"Loading data from {path}")
            # TODO: load feat as csr
            df = pd.read_csv(path, index_col=index_col, **kwargs)
            # Labels: cell x cell-type; Data: feature x cell (need to transpose)
            df = df.T if transpose else df
            # Add dataset info to index
            dataset_name = "_".join(osp.basename(path).split("_")[:-1])
            df.index = dataset_name + "_" + df.index.astype(str)
            dfs.append(df)
        combined_df = pd.concat(dfs)
        return combined_df

    @staticmethod
    def get_map_dict(map_file_path: str, tissue: str) -> Dict[str, Set[str]]:
        """Load cell-type mappings.

        Parameters
        ----------
        map_file_path
            Path to the mapping file.
        tissue
            Tissue of interest.

        Notes
        -----
        Merge mapping across all test sets for the required tissue.

        """
        map_df = pd.read_excel(osp.join(map_file_path, "map.xlsx"))
        map_dict = collections.defaultdict(set)
        for _, row in map_df.iterrows():
            if row["Tissue"] == tissue:
                map_dict[row["Celltype"]].add(row["Training dataset cell type"])
        return dict(map_dict)


@register_dataset("clustering")
class ClusteringDataset(BaseDataset):
    """Data downloading and loading for clustering.

    Parameters
    ----------
    data_dir
        Path to store datasets.
    dataset
        Choice of dataset. Available options are '10X_PBMC', 'mouse_bladder_cell', 'mouse_ES_cell', 'worm_neuron_cell'.

    """

    URL_DICT: Dict[str, str] = {
        "10X_PBMC": "https://www.dropbox.com/s/pfunm27qzgfpj3u/10X_PBMC.h5?dl=1",
        "mouse_bladder_cell": "https://www.dropbox.com/s/xxtnomx5zrifdwi/mouse_bladder_cell.h5?dl=1",
        "mouse_ES_cell": "https://www.dropbox.com/s/zbuku7oznvji8jk/mouse_ES_cell.h5?dl=1",
        "worm_neuron_cell": "https://www.dropbox.com/s/58fkgemi2gcnp2k/worm_neuron_cell.h5?dl=1",
    }
    AVAILABLE_DATA = sorted(URL_DICT)

    def __init__(self, data_dir: str = "./data", dataset: str = "mouse_bladder_cell"):
        super().__init__(data_dir, full_download=False)
        self.data_dir = data_dir
        self.dataset = dataset

    @property
    def data_path(self) -> str:
        return osp.join(self.data_dir, f"{self.dataset}.h5")

    def download(self):
        download_file(self.URL_DICT[self.dataset], self.data_path)

    def is_complete(self):
        return osp.exists(self.data_path)

    def _load_raw_data(self) -> Tuple[ad.AnnData, np.ndarray]:
        with h5py.File(self.data_path, "r") as f:
            x = np.array(f["X"])
            y = np.array(f["Y"])
        adata = ad.AnnData(x, dtype=np.float32)
        return adata, y

    def _raw_to_dance(self, raw_data: Tuple[ad.AnnData, np.ndarray]):
        adata, y = raw_data
        adata.obsm["Group"] = y
        data = Data(adata, train_size="all")
        return data


@register_dataset("imputation")
class ImputationDataset(BaseDataset):

    URL = {
        "pbmc_data": "https://www.dropbox.com/s/brj3orsjbhnhawa/5k.zip?dl=0",
        "mouse_embryo_data": "https://www.dropbox.com/s/8ftx1bydoy7kn6p/GSE65525.zip?dl=0",
        "mouse_brain_data": "https://www.dropbox.com/s/zzpotaayy2i29hk/neuron_10k.zip?dl=0",
        "human_stemcell_data": "https://www.dropbox.com/s/g2qua2j3rqcngn6/GSE75748.zip?dl=0"
    }
    DATASET_TO_FILE = {
        "pbmc_data": "5k_pbmc_protein_v3_filtered_feature_bc_matrix.h5",
        "mouse_embryo_data": [
            osp.join("GSE65525", i)
            for i in [
                "GSM1599494_ES_d0_main.csv",
                "GSM1599497_ES_d2_LIFminus.csv",
                "GSM1599498_ES_d4_LIFminus.csv",
                "GSM1599499_ES_d7_LIFminus.csv",
            ]
        ],
        "mouse_brain_data": "neuron_10k_v3_filtered_feature_bc_matrix.h5",
        "human_stemcell_data": "GSE75748/GSE75748_sc_time_course_ec.csv.gz"
    }  # yapf: disable
    AVAILABLE_DATA = sorted(URL)

    def __init__(self, data_dir="data", dataset="human_stemcell", train_size=0.1):
        super().__init__(data_dir, full_download=False)
        self.data_dir = data_dir
        self.dataset = dataset
        self.train_size = train_size

    def download(self):

        gene_class = ["pbmc_data", "mouse_brain_data", "mouse_embryo_data", "human_stemcell_data"]

        file_name = {
            "pbmc_data": "5k.zip?dl=0",
            "mouse_embryo_data": "GSE65525.zip?dl=0",
            "mouse_brain_data": "neuron_10k.zip?dl=0",
            "human_stemcell_data": "GSE75748.zip?dl=0"
        }

        dl_files = {
            "pbmc_data": "5k_*",
            "mouse_embryo_data": "GSE65525",
            "mouse_brain_data": "neuron*",
            "human_stemcell_data": "GSE75748"
        }

        if sys.platform != 'win32':
            if not osp.exists(self.data_dir):
                os.system("mkdir " + self.data_dir)
            if not osp.exists(self.data_dir + "/train"):
                os.system("mkdir " + self.data_dir + "/train")

            for class_name in gene_class:
                if not any(map(osp.exists, glob(osp.join(self.data_dir, "train", class_name, dl_files[class_name])))):
                    os.system("mkdir " + self.data_dir + "/train/" + class_name)
                    os.system("wget " + self.URL[class_name])  # assumes linux... mac needs to install
                    os.system("unzip " + file_name[class_name])
                    os.system("rm " + file_name[class_name])
                    os.system("mv " + dl_files[class_name] + " " + self.data_dir + "/train/" + class_name + "/")
            os.system("cp -r " + self.data_dir + "/train/ " + self.data_dir + "/test")
        if sys.platform == 'win32':
            if not osp.exists(self.data_dir):
                os.system("mkdir " + self.data_dir)
            if not osp.exists(self.data_dir + "/train"):
                os.mkdir(self.data_dir + "/train")
            for class_name in gene_class:
                if not any(map(osp.exists, glob(osp.join(self.data_dir, "train", class_name, dl_files[class_name])))):
                    os.mkdir(self.data_dir + "/train/" + class_name)
                    os.system("curl " + self.URL[class_name])
                    os.system("tar -xf " + file_name[class_name])
                    os.system("del -R " + file_name[class_name])
                    os.system("move " + dl_files[class_name] + " " + self.data_dir + "/train/" + class_name + "/")
            os.system("copy /r " + self.data_dir + "/train/ " + self.data_dir + "/test")

    def is_complete(self):
        # check whether data is complete or not
        check = [
            self.data_dir + "/train",
            self.data_dir + "/test",
        ]

        for i in check:
            if not osp.exists(i):
                logger.info("file {} doesn't exist".format(i))
                return False
        return True

    def _load_raw_data(self) -> ad.AnnData:
        if self.dataset[-5:] != '_data':
            dataset = self.dataset + '_data'
        else:
            dataset = self.dataset

        if self.dataset == 'mouse_embryo' or self.dataset == 'mouse_embryo_data':
            for i in range(len(self.DATASET_TO_FILE[dataset])):
                fname = self.DATASET_TO_FILE[dataset][i]
                data_path = f'{self.data_dir}/train/{dataset}/{fname}'
                if i == 0:
                    counts = pd.read_csv(data_path, header=None, index_col=0)
                    time = pd.Series(np.zeros(counts.shape[1]))
                else:
                    x = pd.read_csv(data_path, header=None, index_col=0)
                    time = pd.concat([time, pd.Series(np.zeros(x.shape[1])) + i])
                    counts = pd.concat([counts, x], axis=1)
            time = pd.DataFrame(time)
            time.columns = ['time']
            counts = counts.T
            counts.index = [i for i in range(counts.shape[0])]
            adata = ad.AnnData(csr_matrix(counts.values))
            adata.var_names = counts.columns.tolist()
            adata.obs['time'] = time.to_numpy()
        else:
            data_path = osp.join(self.data_dir, "train", dataset, self.DATASET_TO_FILE[dataset])
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"{data_path} does not exist")

            if self.DATASET_TO_FILE[dataset][-3:] == 'csv':
                counts = pd.read_csv(data_path, index_col=0, header=None)
                counts = counts.T
                adata = ad.AnnData(csr_matrix(counts.values))
                # adata.obs_names = ["%d"%i for i in range(adata.shape[0])]
                adata.obs_names = counts.index.tolist()
                adata.var_names = counts.columns.tolist()
            if self.DATASET_TO_FILE[dataset][-2:] == 'gz':
                counts = pd.read_csv(data_path, index_col=0, compression='gzip', header=0)
                counts = counts.T
                adata = ad.AnnData(csr_matrix(counts.values))
                # adata.obs_names = ["%d" % i for i in range(adata.shape[0])]
                adata.obs_names = counts.index.tolist()
                adata.var_names = counts.columns.tolist()
            elif self.DATASET_TO_FILE[dataset][-2:] == 'h5':
                adata = sc.read_10x_h5(data_path)
                adata.var_names_make_unique()

        return adata

    def _raw_to_dance(self, raw_data: ad.AnnData):
        adata = raw_data
        data = Data(adata, train_size=int(adata.n_obs * self.train_size))
        return data
