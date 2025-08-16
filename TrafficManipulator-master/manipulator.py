# from pso import PSO
# from AfterImageExtractor.FEKitsune import Kitsune
# from AfterImageExtractor.KitsuneTools import *
# import numpy as np
# import pickle as pkl
# from scapy.all import *
# from utils import *

# import platform
# import os
# import sys

# import datetime
# from decimal import *
# import argparse

# # statistics for eval

# STA_X_list = []
# STA_feature_list = []
# STA_pktList_list = []
# STA_gbl_dis_list = []
# STA_avg_dis_list = []
# STA_all_feature_list = []


# class Manipulator:

#     # Manipulator Parameters
#     grp_size = 5
#     min_time_extend = 0.
#     max_time_extend = 5.
#     max_cft_pkt = 5
#     max_crafted_pkt_prob = 1.

#     # Particle Parameters
#     w = 0.4
#     c1 = 0.5
#     c2 = 1.

#     # PSO Parameters
#     pso_iter = 10
#     pso_num = 20
#     pso_size = 5

#     # Data Members
#     pktList = []
#     global_FE = None
#     mimic_set = None

#     def __init__(
#             self,
#             mal_pcap_file,
#             mimic_set,
#             knormer_file,
#             init_pcap_file="./data/empty.pcap",  # preparatory traffic
#     ):
#         self.mal_pcap_file = mal_pcap_file

#         print("@Manipulator: Initializing ...")

#         self.mimic_set = np.load(mimic_set)

#         print("self.mimic_set.shape", self.mimic_set.shape)

#         # Normalizer
#         with open(knormer_file, 'rb') as f:
#             self.knormer = pkl.load(f)

#         self.pktList = rdpcap(mal_pcap_file)
#         print("  read %d packets in malicious pcap" % (len(self.pktList)))

#         # Create global feature extractor
#         init_scapy_in = rdpcap(init_pcap_file)
#         self.global_FE = Kitsune(init_scapy_in, np.Inf)

#         # compile preparatory traffic if exists
#         if init_pcap_file != "./data/empty.pcap":
#             RunFE(self.global_FE)

#     def change_manipulator_params(self,
#                                 grp_size=5,
#                                 max_time_extend=5.,
#                                 max_cft_pkt=5,
#                                 min_time_extend=0.,
#                                 max_crafted_pkt_prob=1.):
#         self.grp_size = grp_size
#         self.max_time_extend = max_time_extend
#         self.max_cft_pkt = max_cft_pkt
#         self.min_time_extend = min_time_extend
#         self.max_crafted_pkt_prob = max_crafted_pkt_prob

#     def change_pso_params(self, max_iter=10, particle_num=20, grp_size=5):
#         self.pso_iter = max_iter
#         self.pso_num = particle_num
#         self.pso_size = grp_size

#     def change_particle_params(self, w=0.4, c1=0.5, c2=1.):
#         self.w = w
#         self.c1 = c1
#         self.c2 = c2

#     def save_configurations(self, config_file):

#         print("@Mani: Save configurations...")
#         with open(config_file, "w") as f:
#             f.write("+----Highlight----+\r\n")
#             f.write('(iter,swarm,delay,mimic) (' + str(self.pso_iter) + ',' +
#                     str(self.pso_num) + ',' + str(self.max_time_extend) + ',' +
#                     str(len(self.mimic_set)) + ")\r\n")
#             f.write("-" * 96 + "\r\n")
#             f.write("Feature extractor: AfterImage\r\n")
#             f.write("-" * 96 + "\r\n")
#             f.write("(Manipulator Params)\r\n")
#             f.write("  grp_size:        " + str(self.grp_size) + "\r\n")
#             f.write("  min_time_extend: " + str(self.min_time_extend) + "\r\n")
#             f.write("  max_time_extend: " + str(self.max_time_extend) + "\r\n")
#             f.write("  max_cft_pkt:     " + str(self.max_cft_pkt) + "\r\n")
#             f.write("  min_cft_pkt_prob:" + str(0) + "\r\n")
#             f.write("  max_cft_pkt_prob:" + str(self.max_crafted_pkt_prob) +
#                     "\r\n\r\n")
#             f.write("(PSO Params)\r\n")
#             f.write("  pso_iter:        " + str(self.pso_iter) + "\r\n")
#             f.write("  pso_num:         " + str(self.pso_num) + "\r\n")
#             f.write("  pso_size:        " + str(self.pso_size) + "\r\n\r\n")
#             f.write("(Particle Params)\r\n")
#             f.write("  w:               " + str(self.w) + "\r\n")
#             f.write("  c1:              " + str(self.c1) + "\r\n")
#             f.write("  c2:              " + str(self.c2) + "\r\n")
#             f.write("-" * 96 + "\r\n")

#     def process(
#         self,
#         sta_file,
#         start_no=0,
#         limit=np.Inf,
#         heuristic=False,
#     ):

#         # Timers
#         FE_time = 0
#         import time
#         timer = time.time()

#         acc_ics_time = 0
#         last_end_time = float(self.pktList[0].time)
#         begin_timestamp = float(self.pktList[0].time)

#         st = start_no
#         ed = st + self.grp_size

#         print("@Mani: Begin processing...")
#         while True:
#             print("@Manipulator: Processing pkt ( %d to %d ) ..." % (st, ed))

#             # print("@Manipulator: Create PSO")
#             # ---- initialize PSO--------------------------------------------+
#             pso = PSO(max_iter=self.pso_iter,
#                     particle_num=self.pso_num,
#                     grp_size=self.pso_size)

#             # ---- load a new pkt group--------------------------------------+
#             groupList = self.pktList[st:ed]

#             # ---- increase initial time of the new pkt group----------------+
#             for pkt in groupList:
#                 # pkt.time += Decimal(acc_ics_time)
#                 # pkt.time += acc_ics_time
#                 pkt.time = float(pkt.time) + acc_ics_time

#             # ---- execute PSO-----------------------------------------------+
#             pso_show_info = True
#             if self.grp_size < 50:
#                 pso_show_info = False
#             ics_time, cur_end_time, \
#             STA_best_X, STA_best_feature, STA_best_pktList, STA_gbl_dis, STA_avg_dis,STA_best_all_feature,fe_time\
#                         = pso.execute(  last_end_time, groupList,
#                                         self.max_time_extend,self.max_cft_pkt, self.min_time_extend, self.max_crafted_pkt_prob,
#                                         self.mimic_set,
#                                         self.global_FE.FE.nstat,
#                                         self.knormer,
#                                         self.w,self.c1,self.c2,
#                                         pso_show_info,heuristic)

#             FE_time += fe_time
#             # ---- prepare for next pkt group--------------------------------+
#             acc_ics_time += ics_time
#             last_end_time = cur_end_time

#             ttime1 = time.perf_counter()
#             nstat = self.global_FE.FE.nstat
#             self.global_FE = Kitsune(STA_best_pktList, np.Inf, False)
#             self.global_FE.FE.nstat = safelyCopyNstat(nstat, False)
#             RunFE(self.global_FE)
#             ttime2 = time.perf_counter()
#             FE_time += (ttime2 - ttime1)

#             # ---- Update statistics ----------------------------------------------+
#             global STA_X_list
#             global STA_feature_list
#             global STA_pktList_list
#             global STA_gbl_dis_list
#             global STA_avg_dis_list

#             STA_X_list.append(STA_best_X)
#             STA_feature_list.append(STA_best_feature)
#             STA_pktList_list.append(STA_best_pktList)
#             STA_gbl_dis_list.append(STA_gbl_dis)
#             STA_avg_dis_list.append(STA_avg_dis)
#             STA_all_feature_list.append(STA_best_all_feature)

# # ---my function to save the crafted packets---------------------+

#             if st != 0 and (st % 1000 == 0 or ed == len(self.pktList)) or ed == limit:
#                 print("@Manipulator:Time elapsed:", time.time() - timer)
#                 with open(sta_file, "wb") as f:
#                     pkl.dump(STA_X_list, f)
#                     pkl.dump(STA_feature_list, f)
#                     pkl.dump(STA_pktList_list, f)
#                     pkl.dump(STA_gbl_dis_list, f)
#                     pkl.dump(STA_avg_dis_list, f)
#                     pkl.dump(STA_all_feature_list, f)
#                 print("@Manipulator:statistics.pkl is updated...")  
#                 # Save manipulated packets to a pcap file 
#                 all_packets = [pkt for group in STA_pktList_list for pkt in group]
#                 base_name, ext = os.path.splitext(self.mal_pcap_file)
#                 wrpcap(os.path.join(base_name + "_manipulated.pcap"), all_packets)
#                 print("@Manipulator:Manipulated packets are saved to " + base_name + "_manipulated.pcap")

#             # ---plt and dump info-------------------------------------------+
#             if st != 0 and (st % 1000 == 0
#                             or ed == len(self.pktList)) or ed == limit:
#                 print("@Manipulator:Time elapsed:", time.time() - timer)
#                 with open(sta_file, "wb") as f:
#                     pkl.dump(STA_X_list, f)
#                     pkl.dump(STA_feature_list, f)
#                     pkl.dump(STA_pktList_list, f)
#                     pkl.dump(STA_gbl_dis_list, f)
#                     pkl.dump(STA_avg_dis_list, f)
#                     pkl.dump(STA_all_feature_list, f)
#                 print("@Manipulator:statistics.pkl is updated...")

#             # ---------------update `st` and `ed` for next loop--------------+
#             if ed == len(self.pktList) or ed == limit:
#                 print("@Manipulator:All Finished!", ed,
#                       "Pkts Processed,Time elapsed:",
#                       time.time() - timer, "FE_time:", FE_time)
#                 break

#             st = ed
#             ed += self.grp_size
#             if ed >= len(self.pktList):
#                 ed = len(self.pktList)
#                 self.grp_size = ed - st
#             if ed >= limit:
#                 ed = limit
#                 self.grp_size = ed - st



# # python main.py -m example/test.pcap -b example/mimic_set.npy -n example/normalizer.pkl -i example/init.pcap



import os
import time
import pickle as pkl
import numpy as np
from scapy.all import rdpcap, wrpcap
from pso import PSO
from AfterImageExtractor.FEKitsune import Kitsune
from AfterImageExtractor.KitsuneTools import RunFE, safelyCopyNstat

class Manipulator:
    def __init__(
        self,
        mal_pcap_file,
        mimic_set_file,
        knormer_file,
        init_pcap_file="./data/empty.pcap",
    ):
        # File paths
        self.mal_pcap_file = mal_pcap_file
        base_name, _ = os.path.splitext(mal_pcap_file)
        self.stats_file = f"{base_name}_statistics.pkl"
        self.output_pcap = f"{base_name}_manipulated.pcap"

        # Load mimic set and normalizer
        print(f"@Manipulator: Initializing for {mal_pcap_file}")
        self.mimic_set = np.load(mimic_set_file)
        print(f"  mimic_set shape: {self.mimic_set.shape}")
        with open(knormer_file, 'rb') as f:
            self.knormer = pkl.load(f)

        # Read packets
        self.pktList = rdpcap(mal_pcap_file)
        print(f"  Loaded {len(self.pktList)} packets")

        # Global feature extractor setup
        init_pkts = rdpcap(init_pcap_file)
        self.global_FE = Kitsune(init_pkts, np.Inf)
        if init_pcap_file != "./data/empty.pcap":
            RunFE(self.global_FE)

        # Default parameters
        self.grp_size = 5
        self.min_time_extend = 0.0
        self.max_time_extend = 5.0
        self.max_cft_pkt = 5
        self.max_crafted_pkt_prob = 1.0
        self.pso_iter = 10
        self.pso_num = 20
        self.pso_size = 5
        self.w = 0.4
        self.c1 = 0.5
        self.c2 = 1.0

        # Statistics containers
        self.STA_X_list = []
        self.STA_feature_list = []
        self.STA_pktList_list = []
        self.STA_gbl_dis_list = []
        self.STA_avg_dis_list = []
        self.STA_all_feature_list = []

    def set_manipulator_params(
        self, grp_size=5, min_time_extend=0.0, max_time_extend=5.0,
        max_cft_pkt=5, max_crafted_pkt_prob=1.0
    ):
        self.grp_size = grp_size
        self.min_time_extend = min_time_extend
        self.max_time_extend = max_time_extend
        self.max_cft_pkt = max_cft_pkt
        self.max_crafted_pkt_prob = max_crafted_pkt_prob

    def set_pso_params(self, max_iter=10, particle_num=20, grp_size=5):
        self.pso_iter = max_iter
        self.pso_num = particle_num
        self.pso_size = grp_size

    def set_particle_params(self, w=0.4, c1=0.5, c2=1.0):
        self.w = w
        self.c1 = c1
        self.c2 = c2

    def process(self, start_no=0, limit=None, heuristic=False):
        if limit is None:
            limit = len(self.pktList)

        last_end_time = float(self.pktList[0].time)
        acc_ics_time = 0.0
        st = start_no
        ed = min(st + self.grp_size, limit)
        timer_start = time.time()
        FE_time = 0.0

        print("@Manipulator: Begin processing...")
        while st < ed:
            print(f"Processing packets {st} to {ed}...")
            # Initialize PSO
            pso = PSO(
                max_iter=self.pso_iter,
                particle_num=self.pso_num,
                grp_size=self.pso_size
            )

            # Prepare packet group
            group = self.pktList[st:ed]
            for pkt in group:
                pkt.time = float(pkt.time) + acc_ics_time

            # Execute PSO
            results = pso.execute(
                last_end_time, group,
                self.max_time_extend, self.max_cft_pkt,
                self.min_time_extend, self.max_crafted_pkt_prob,
                self.mimic_set, self.global_FE.FE.nstat,
                self.knormer, self.w, self.c1, self.c2,
                show_info=(self.grp_size >= 50),
                heuristic=heuristic
            )
            (
                ics_time, cur_end_time,
                best_X, best_feat, best_pkts,
                gbl_dis, avg_dis, all_feat, fe_time
            ) = results
            FE_time += fe_time

            # Update global feature extractor
            acc_ics_time += ics_time
            last_end_time = cur_end_time
            saved_nstat = self.global_FE.FE.nstat
            self.global_FE = Kitsune(best_pkts, np.Inf, False)
            self.global_FE.FE.nstat = safelyCopyNstat(saved_nstat, False)
            start_fe = time.perf_counter()
            RunFE(self.global_FE)
            FE_time += (time.perf_counter() - start_fe)

            # Collect statistics
            self.STA_X_list.append(best_X)
            self.STA_feature_list.append(best_feat)
            self.STA_pktList_list.append(best_pkts)
            self.STA_gbl_dis_list.append(gbl_dis)
            self.STA_avg_dis_list.append(avg_dis)
            self.STA_all_feature_list.append(all_feat)

            # Periodic save
            if (st != 0 and (st % 1000 == 0 or ed == limit)):
                elapsed = time.time() - timer_start
                print(f"Saving stats after {st} packets, elapsed: {elapsed:.2f}s")
                self._save_stats()

            # Move window
            st = ed
            ed = min(st + self.grp_size, limit)

        print(f"Processing complete. Total time: {time.time() - timer_start:.2f}s, FE_time: {FE_time:.2f}s")
        # Final save
        self._save_stats()
        # Write manipulated pcap
        all_pkts = [pkt for grp in self.STA_pktList_list for pkt in grp]
        wrpcap(self.output_pcap, all_pkts)
        print(f"Manipulated packets written to {self.output_pcap}")

    def _save_stats(self):
        with open(self.stats_file, "wb") as f:
            pkl.dump(self.STA_X_list, f)
            pkl.dump(self.STA_feature_list, f)
            pkl.dump(self.STA_pktList_list, f)
            pkl.dump(self.STA_gbl_dis_list, f)
            pkl.dump(self.STA_avg_dis_list, f)
            pkl.dump(self.STA_all_feature_list, f)
        print(f"Statistics saved to {self.stats_file}")
