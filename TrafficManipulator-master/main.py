# import numpy as np
# import pickle as pkl
# from scapy.all import *
# import argparse
# from manipulator import Manipulator

# parse = argparse.ArgumentParser()
# parse.add_argument('-m',
#                    '--mal_pcap',
#                    type=str,
#                    required=True,
#                    help="input malicious traffic (.pcap)")

# parse.add_argument('-b',
#                    '--mimic_set',
#                    type=str,
#                    required=True,
#                    help="benign features to mimic (.npy)")

# parse.add_argument('-n',
#                    '--normalizer',
#                    type=str,
#                    required=True,
#                    help="compiled feature normalizer (.pkl)")

# parse.add_argument('-i',
#                    '--init_pcap',
#                    type=str,
#                    default='./_empty.pcap',
#                    help="preparatory traffic (ignore this if you don't need)")

# parse.add_argument('-o',
#                    '--sta_file',
#                    type=str,
#                    default='./example/statistics.pkl',
#                    help="file saving the final statistics (.pkl)")

# arg = parse.parse_args()

# m = Manipulator(arg.mal_pcap, arg.mimic_set, arg.normalizer, arg.init_pcap)

# max_iter, particle_num, local_grp_size = 3, 6, 3
# # max_iter,particle_num,local_grp_size = 4,8,4
# # max_iter,particle_num,local_grp_size = 5,10,5
# # max_iter,particle_num,local_grp_size = 3,10,5

# m.change_particle_params(w=0.7298, c1=1.49618, c2=1.49618)
# m.change_pso_params(max_iter=max_iter,
#                     particle_num=particle_num,
#                     grp_size=local_grp_size)
# m.change_manipulator_params(grp_size=100,
#                             min_time_extend=3.,
#                             max_time_extend=6.,
#                             max_cft_pkt=1,
#                             max_crafted_pkt_prob=0.01)

# # m.save_configurations('./configurations.txt')

# # tmp_pcap_file = "_crafted.pcap"
# # m.process(tmp_pcap_file, arg.sta_file, limit=20)

# m.process(arg.sta_file, limit=np.Inf, heuristic=False)

import numpy as np
import argparse
from manipulator import Manipulator

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run packet manipulation and auto-save stats per-pcap."
    )
    parser.add_argument('-m', '--mal_pcap', type=str, required=True,
                        help="Input malicious traffic (.pcap)")
    parser.add_argument('-b', '--mimic_set', type=str, required=True,
                        help="Benign features to mimic (.npy)")
    parser.add_argument('-n', '--normalizer', type=str, required=True,
                        help="Compiled feature normalizer (.pkl)")
    parser.add_argument('-i', '--init_pcap', type=str, default='./_empty.pcap',
                        help="Preparatory traffic (optional)")
    parser.add_argument('--limit', type=int, default=None,
                        help="Max number of packets to process (default: all)")
    parser.add_argument('--heuristic', action='store_true',
                        help="Enable heuristic mode in PSO")

    args = parser.parse_args()

    # Initialize manipulator (auto-generates stats and output pcap names)
    m = Manipulator(
        mal_pcap_file=args.mal_pcap,
        mimic_set_file=args.mimic_set,
        knormer_file=args.normalizer,
        init_pcap_file=args.init_pcap
    )

    # Tuning PSO & manipulator parameters
    max_iter = 3
    particle_num = 6
    local_grp_size = 3
    m.set_particle_params(w=0.7298, c1=1.49618, c2=1.49618)
    m.set_pso_params(max_iter=max_iter,
                    particle_num=particle_num,
                    grp_size=local_grp_size)
    m.set_manipulator_params(
        grp_size=100,
        min_time_extend=3.0,
        max_time_extend=6.0,
        max_cft_pkt=1,
        max_crafted_pkt_prob=0.01
    )

    # Run processing (stats written to <mal_pcap>_statistics.pkl)
    m.process(
        start_no=0,
        limit=args.limit,
        heuristic=args.heuristic
    )