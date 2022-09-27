import shelve
import sfcsim
network_index = 3
shelve_file = shelve.open("./network_file/network")
network_name = "cernnet2_" + str(network_index)
network = sfcsim.cernnet2_sfc_dynamic(num_sfc = 1000,network_name = network_name)
shelve_file[network_name] = network
shelve_file.close()
