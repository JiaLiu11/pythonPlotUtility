#! /usr/bin/env python

from sys import argv, exit
from os import path, listdir
import numpy as np
import re


class ParticleFilter:
    """
        This class contains functions to filter out interesting particles
        defined by the user from the UrQMD or OSCAR outputs
    """

    def __init__(self, enable_hdf5=True, enable_sqlite=True):
        self.enable_hdf5 = enable_hdf5
        self.enable_sqlite = enable_sqlite
        self.output_filename = "particles"
        self.flow_analyze_order = 7
        self.p_t_min = 0.0
        self.p_t_max = 3.0
        self.n_p_t = 30

        # particle_name, pid
        self.pid_dict = {
            "total": 0,
            "charged": 1,
            "charged_eta": 2,
            "pion": 6,  # sum(7, 8, -7)
            "pion_p": 7,
            "pion_0": 8,
            "pion_m": -7,
            "kaon": 11,  # sum(12, 13)
            "kaon_p": 12,
            "kaon_0": 13,
            "anti_kaon": -11,  # sum(-12, -13)
            "kaon_m": -12,
            "anti_kaon_0": -13,
            "nucleon": 16,  # sum(17, 18)
            "proton": 17,
            "neutron": 18,
            "anti_nucleon": -16,  # sum(-17, -18)
            "anti_proton": -17,
            "anti_neutron": -18,
            "sigma": 21,  # sum(22, 23, 24)
            "sigma_p": 22,
            "sigma_0": 23,
            "sigma_m": 24,
            "anti_sigma": -21,
            "anti_sigma_p": -22,
            "anti_sigma_0": -23,
            "anti_sigma_m": -24,
            "xi": 26,  # sum(27, 28)
            "xi_0": 27,
            "xi_m": 28,
            "anti_xi": -26,
            "anti_xi_0": -27,
            "anti_xi_m": -28,
            "lambda": 31,
            "anti_lambda": -31,
            "omega": 36,
            "anti_omega": -36,
            "phi": 41,
            "rho": 46,  # sum(47, 48, -47)
            "rho_p": 47,
            "rho_0": 48,
            "rho_m": -47,
            "eta": 51,
            "eta_prime": 52,
            "gamma": 61,
            "omega782": 65,
        }

        for aparticle in self.pid_dict.keys():
            if self.pid_dict[aparticle] >= 0:
                self.pid_dict[aparticle + "_hydro"] = (
                    self.pid_dict[aparticle] + 1000)
            else:
                self.pid_dict[aparticle + "_hydro"] = (
                    self.pid_dict[aparticle] - 1000)
            if self.pid_dict[aparticle] >= 0:
                self.pid_dict[aparticle + "_thermal"] = (
                    self.pid_dict[aparticle] + 2000)
            else:
                self.pid_dict[aparticle + "_thermal"] = (
                    self.pid_dict[aparticle] - 2000)

        self.pid_dict.update({
            "photon_total": 9000,
            "photon_total_eq": 9001,
            "photon_QGP_tot": 9002,
            "photon_QGP_eq": 9003,
            "photon_HG_tot": 9004,
            "photon_HG_eq": 9005,
            "direct_gamma_shortdecay_hydro": 9006,
            "decay_gamma_pi0_hydro": 9007,
            "decay_gamma_eta_hydro": 9008,
            "decay_gamma_omega_hydro": 9009,
            "decay_gamma_phi_hydro": 9010,
            "decay_gamma_etap_hydro": 9011,
            "decay_gamma_Sigma0_hydro": 9012,
        })

        # UrQMD pid Dictionary, name conversion defined as in binUtility
        # particle name, UrQMD id# : isospin*2000 + pid
        self.urqmd_pid_dict = {
            2101: "pion_p",
            -1899: "pion_m",
            101: "pion_0",
            1106: "kaon_p",
            -894: "kaon_0",
            -1106: "kaon_m",
            894: "anti_kaon_0",
            1001: "proton",
            -999: "neutron",
            -1001: "anti_proton",
            999: "anti_neutron",
            2040: "sigma_p",
            -1960: "sigma_m",
            40: "sigma_0",
            -2040: "anti_sigma_p",
            1960: "anti_sigma_m",
            -40: "anti_sigma_0",
            1049: "xi_0",
            -951: "xi_m",
            -1049: "anti_xi_0",
            951: "anti_xi_m",
            27: "lambda",
            -27: "anti_lambda",
            55: "omega",
            -55: "anti_omega",
            109: "phi",
            102: "eta",
            107: "eta_prime",
            100: "gamma",
        }

        # pdg pid Dictionary
        # pdg id#, particle name
        self.pdg_pid_dict = {
            211: "pion_p",
            -211: "pion_m",
            111: "pion_0",
            321: "kaon_p",
            311: "kaon_0",
            -321: "kaon_m",
            -311: "anti_kaon_0",
            2212: "proton",
            2112: "neutron",
            -2212: "anti_proton",
            -2112: "anti_neutron",
            3222: "sigma_p",
            3112: "sigma_m",
            3212: "sigma_0",
            -3222: "anti_sigma_p",
            -3112: "anti_sigma_m",
            -3212: "anti_sigma_0",
            3322: "xi_0",
            3312: "xi_m",
            -3322: "anti_xi_0",
            -3312: "anti_xi_m",
            3122: "lambda",
            -3122: "anti_lambda",
            3334: "omega",
            -3334: "anti_omega",
            333: "phi",
            221: "eta",
            331: "eta_prime",
            22: "gamma",
        }

        #particle mass Dictionary (unit in GeV)
        self.mass_pid_dict = {
            "pion": 0.13957,
            "pion_p": 0.13957,
            "pion_0": 0.13498,
            "pion_m": 0.13957,
            "kaon": 0.49368,
            "kaon_p": 0.49368,
            "kaon_0": 0.49765,
            "anti_kaon": 0.49368,
            "kaon_m": 0.49368,
            "anti_kaon_0": 0.49765,
            "nucleon": 0.93827,
            "proton": 0.93827,
            "neutron": 0.93957,
            "anti_nucleon": 0.93827,
            "anti_proton": 0.93827,
            "anit_neutron": 0.93957,
            "sigma": 1.18937,
            "sigma_p": 1.18937,
            "sigma_0": 1.19264,
            "sigma_m": 1.19745,
            "anti_sigma": 1.18937,
            "anti_sigma_p": 1.18937,
            "anti_sigma_0": 1.19264,
            "anti_sigma_m": 1.19745,
            "xi": 1.31483,
            "xi_0": 1.31483,
            "xi_m": 1.32131,
            "anti_xi": 1.31483,
            "anti_xi_0": 1.31483,
            "anti_xi_m": 1.32131,
            "lambda": 1.11568,
            "anti_lambda": 1.11568,
            "omega": 1.67243,
            "anti_omega": 1.67243,
            "rho": 0.77580,
            "rho_p": 0.77580,
            "rho_0": 0.77580,
            "rho_m": 0.77580,
            "phi": 1.01946,
            "eta": 0.54775,
            "eta_prime": 0.95778,
            "gamma": 0.0,
        }
        for aparticle in self.mass_pid_dict.keys():
            self.mass_pid_dict[aparticle + "_hydro"] = (
                self.mass_pid_dict[aparticle])
            self.mass_pid_dict[aparticle + "_thermal"] = (
                self.mass_pid_dict[aparticle])

        # charged hadrons list
        self.charged_hadron_list = [
            "pion_p", "pion_m", "kaon_p", "kaon_m", "proton", "anti_proton",
            "sigma_p", "sigma_m", "anti_sigma_p", "anti_sigma_m",
            "xi_m", "anti_xi_m"]

    def collect_particles_oscar(self, folder, hydro_event_id, result_filename,
                                particles_to_collect, rap_type, rap_range):
        """
            This function collects particles momentum and space-time
            information from OSCAR format output file "result_filename" into
            database for one hydro event with hydro_event_id.
            It assigns each UrQMD run an additional urqmd_event_id.
        """
        if self.enable_hdf5:
            import h5py

            of_hdf5 = h5py.File("%s.hdf5" % self.output_filename, 'w')
            hydro_event_group = (
                of_hdf5.create_group("hydro_event_%d" % int(hydro_event_id)))
        if self.enable_sqlite:
            from DBR import SqliteDB

            of_db = SqliteDB("%s.db" % self.output_filename)
            # first write the pid_lookup table, makes sure there is only one
            # such table
            if of_db.createTableIfNotExists(
                    "pid_lookup", (("name", "text"), ("pid", "integer"))):
                of_db.insertIntoTable(
                    "pid_lookup", list(self.pid_dict.items()))
            # write the particle mass table
            if of_db.createTableIfNotExists(
                    "pid_Mass", (("name", "text"), ("mass", "real"))):
                of_db.insertIntoTable(
                    "pid_Mass", list(self.mass_pid_dict.items()))
            # create tables
            of_db.createTableIfNotExists(
                "particle_list", (
                    ("hydroEvent_id", "integer"),
                    ("UrQMDEvent_id", "interger"), ("pid", "integer"),
                    ("tau", "real"), ("x", "real"), ("y", "real"),
                    ("eta", "real"), ("pT", "real"), ("phi_p", "real"),
                    ("rapidity", "real"), ("pseudorapidity", "real"))
            )

        pid_to_collect = []
        for aparticle in particles_to_collect:
            if aparticle == "charged":
                pid_to_collect += (
                    map(lambda x: self.pid_dict[x], self.charged_hadron_list))
            else:
                pid_to_collect += [self.pid_dict[aparticle]]

        # check input file
        urqmd_outputfile = path.join(folder, result_filename)
        if not path.isfile(urqmd_outputfile):
            exit("Cannot find OSCAR output file: " + urqmd_outputfile)

        # read in OSCAR outputs and fill them into database
        # this routine is copied and modified from binUtility
        read_mode = "header_first_part"
        # the first read line is already part of the header line
        header_count = 1
        data_row_count = 0
        urqmd_event_id = 1
        for aLine in open(urqmd_outputfile):
            if read_mode == "header_first_part":
                if header_count < 3:  # skip first 3 lines
                    header_count += 1
                    continue
                read_mode = "header_second_part"
            elif read_mode == "header_second_part":
                try:
                    data_row_count = int(aLine.split()[1])
                    phi_rotation = np.random.uniform(0, 2 * np.pi)
                    particle_info = []
                except ValueError as e:
                    print(
                        "The file " + urqmd_outputfile +
                        " does not have a valid OSCAR output file header!")
                    exit(e)
                read_mode = "data_part"
            elif read_mode == "data_part":
                if data_row_count > 0:
                    # have data to read
                    try:
                        px, py, pz, p0 = (
                            map(lambda x: float(x.replace("D", "E")),
                                aLine[25:126].split()))
                        x, y, z, t = (
                            map(lambda x: float(x.replace("D", "E")),
                                aLine[155:256].split()))
                        pdg_id = int(aLine[15:22])
                        try:
                            database_pid = (
                                self.pid_dict[self.pdg_pid_dict[pdg_id]])
                        except ValueError as e:
                            print(
                                "Can not find particle id in the dictionary!")
                            exit(e)
                        if database_pid in pid_to_collect:
                            p_mag = np.sqrt(px * px + py * py + pz * pz)
                            rap = 0.5 * np.log((p0 + pz) / (p0 - pz))
                            pseudorap = (
                                0.5 * np.log((p_mag + pz) / (p_mag - pz)))
                            if rap_type == 'rapidity':
                                rap_judge = rap
                            else:
                                rap_judge = pseudorap
                            if rap_range[1] > rap_judge > rap_range[0]:
                                p_t = np.sqrt(px * px + py * py)
                                x_rotated = (
                                    x * np.cos(phi_rotation)
                                    - y * np.sin(phi_rotation))
                                y_rotated = (
                                    y * np.cos(phi_rotation)
                                    + x * np.sin(phi_rotation))
                                px_rotated = (
                                    px * np.cos(phi_rotation)
                                    - py * np.sin(phi_rotation))
                                py_rotated = (
                                    py * np.cos(phi_rotation)
                                    + px * np.sin(phi_rotation))
                                phi = np.arctan2(py_rotated, px_rotated)

                                tau = np.sqrt(t * t - z * z)
                                eta = 0.5 * np.log((t + z) / (t - z))
                                particle_info.append(
                                    [pdg_id, tau, x_rotated, y_rotated, eta,
                                     p_t, phi, rap, pseudorap])
                                # output data for sqlite database
                                if self.enable_sqlite:
                                    of_db.insertIntoTable(
                                        "particle_list",
                                        (hydro_event_id, urqmd_event_id,
                                         database_pid, float(tau),
                                         float(x_rotated), float(y_rotated),
                                         float(eta), float(p_t), float(phi),
                                         float(rap), float(pseudorap))
                                    )
                    except ValueError as e:
                        print(
                            "The file " + urqmd_outputfile +
                            " does not have valid OSCAR data!")
                        exit(e)
                    data_row_count -= 1
                if data_row_count == 0:
                    # store data
                    particle_info = np.asarray(particle_info)
                    if self.enable_hdf5:
                        urqmd_event_group = (hydro_event_group.create_group(
                            "urqmd_event_%d" % urqmd_event_id))
                        pid_list = list(set(list(particle_info[:, 0])))
                        for ipid in range(len(pid_list)):
                            particle_name = self.pdg_pid_dict[pid_list[ipid]]
                            particle_group = urqmd_event_group.create_group(
                                "%s" % particle_name)
                            particle_group.attrs.create(
                                'mass', self.mass_pid_dict[particle_name])
                            idx = particle_info[:, 0] == pid_list[ipid]
                            particle_data = particle_info[idx, 1:9]
                            n_particle = len(particle_data[:, 0])
                            particle_group.attrs.create(
                                'N_particle', n_particle)
                            particle_group.create_dataset(
                                "particle_info", data=particle_data,
                                compression='gzip', compression_opts=9)
                    if urqmd_event_id % 100 == 0:
                        print("processing OSCAR events %d finished."
                              % urqmd_event_id)
                    urqmd_event_id += 1
                    # switch back to header mode
                    data_row_count = 0
                    header_count = 0  # not pointing at the header line yet
                    read_mode = "header_second_part"

        if self.enable_hdf5:
            of_hdf5.flush()
            of_hdf5.close()
        # close connection to commit changes
        if self.enable_sqlite:
            of_db.closeConnection()

    def collect_particles_urqmd(self, folder, hydro_event_id, result_filename,
                                particles_to_collect, rap_type, rap_range):
        """
            This function collects particles momentum and space-time
            information from UrQMD format output file "result_filename" into
            database for one hydro event with hydro_event_id.
            It assigns each UrQMD run an additional urqmd_event_id.
        """
        if self.enable_hdf5:
            import h5py

            of_hdf5 = h5py.File("%s.hdf5" % self.output_filename, 'w')
            hydro_event_group = (
                of_hdf5.create_group("hydro_event_%d" % int(hydro_event_id)))
        if self.enable_sqlite:
            from DBR import SqliteDB

            of_db = SqliteDB("%s.db" % self.output_filename)
            # first write the pid_lookup table, makes sure there is only one
            # such table
            if of_db.createTableIfNotExists(
                    "pid_lookup", (("name", "text"), ("pid", "integer"))):
                of_db.insertIntoTable(
                    "pid_lookup", list(self.pid_dict.items()))
            # write the particle mass table
            if of_db.createTableIfNotExists(
                    "pid_Mass", (("name", "text"), ("mass", "real"))):
                of_db.insertIntoTable(
                    "pid_Mass", list(self.mass_pid_dict.items()))
            # create tables
            of_db.createTableIfNotExists(
                "particle_list", (
                    ("hydroEvent_id", "integer"),
                    ("UrQMDEvent_id", "interger"), ("pid", "integer"),
                    ("tau", "real"), ("x", "real"), ("y", "real"),
                    ("eta", "real"), ("pT", "real"), ("phi_p", "real"),
                    ("rapidity", "real"), ("pseudorapidity", "real"))
            )

        pid_to_collect = []
        for aparticle in particles_to_collect:
            if aparticle == "charged":
                pid_to_collect += (
                    map(lambda x: self.pid_dict[x], self.charged_hadron_list))
            else:
                pid_to_collect += [self.pid_dict[aparticle]]

        # check input file
        urqmd_outputfile = path.join(folder, result_filename)
        if not path.isfile(urqmd_outputfile):
            exit("Cannot find UrQMD output file: " + urqmd_outputfile)

        # convert UrQMD outputs and fill them into database
        # this routine is copied and modified from binUtility
        read_mode = "header_first_part"
        # the first read line is already part of the header line
        header_count = 1
        data_row_count = 0
        urqmd_event_id = 1
        for aLine in open(urqmd_outputfile):
            if read_mode == "header_first_part":
                if header_count <= 14:  # skip first 14 lines
                    header_count += 1
                    continue
                # now at 15th line
                assert header_count == 15, "No no no... Stop here."
                try:
                    data_row_count = int(aLine.split()[0])
                except ValueError as e:
                    print(
                        "The file " + urqmd_outputfile +
                        " does not have a valid UrQMD output file header!")
                    exit(e)
                # perform a random rotation of the each event
                phi_rotation = np.random.uniform(0, 2 * np.pi)
                particle_info = []

                read_mode = "header_second_part"
            elif read_mode == "header_second_part":
                # skip current line by switching to data reading mode
                read_mode = "data_part"
            elif read_mode == "data_part":
                if data_row_count > 0:
                    # have data to read
                    try:
                        p0, px, py, pz = map(
                            lambda x: float(x.replace("D", "E")),
                            aLine[98:193].split())
                        t, x, y, z = map(lambda x: float(x.replace("D", "E")),
                                         aLine[245:338].split())
                        isospin2 = int(aLine[222:224])
                        pid = int(aLine[216:222])
                        urqmd_pid = pid + isospin2 * 1000
                        try:
                            if pid == 100 and isospin2 != 0:
                                # UrQMD seems to have a bug for decay photon
                                # isospin and charge
                                print(
                                    "Warning: decay photon's isospin is "
                                    "not correct!")
                                urqmd_pid = 100
                            database_pid = self.pid_dict[
                                self.urqmd_pid_dict[urqmd_pid]]
                        except ValueError as e:
                            print(
                                "Can not find particle id in the dictionary!")
                            exit(e)
                        if database_pid in pid_to_collect:
                            p_mag = np.sqrt(px * px + py * py + pz * pz)
                            rap = 0.5 * np.log((p0 + pz) / (p0 - pz))
                            pseudorap = (
                                0.5 * np.log((p_mag + pz) / (p_mag - pz)))
                            if rap_type == 'rapidity':
                                rap_judge = rap
                            else:
                                rap_judge = pseudorap
                            if rap_range[1] > rap_judge > rap_range[0]:
                                p_t = np.sqrt(px * px + py * py)
                                x_rotated = (
                                    x * np.cos(phi_rotation)
                                    - y * np.sin(phi_rotation))
                                y_rotated = (
                                    y * np.cos(phi_rotation)
                                    + x * np.sin(phi_rotation))
                                px_rotated = (
                                    px * np.cos(phi_rotation)
                                    - py * np.sin(phi_rotation))
                                py_rotated = (
                                    py * np.cos(phi_rotation)
                                    + px * np.sin(phi_rotation))
                                phi = np.arctan2(py_rotated, px_rotated)

                                tau = np.sqrt(t * t - z * z)
                                eta = 0.5 * np.log((t + z) / (t - z))
                                particle_info.append(
                                    [urqmd_pid, tau, x_rotated, y_rotated, eta,
                                     p_t, phi, rap, pseudorap])
                                # output data for sqlite database
                                if self.enable_sqlite:
                                    of_db.insertIntoTable(
                                        "particle_list",
                                        (hydro_event_id, urqmd_event_id,
                                         database_pid, float(tau),
                                         float(x_rotated), float(y_rotated),
                                         float(eta), float(p_t), float(phi),
                                         float(rap), float(pseudorap))
                                    )
                    except ValueError as e:
                        print(
                            "The file " + urqmd_outputfile +
                            " does not have valid UrQMD data!")
                        exit(e)
                    data_row_count -= 1
                if data_row_count == 1:
                    particle_info = np.asarray(particle_info)
                    if self.enable_hdf5:
                        urqmd_event_group = (hydro_event_group.create_group(
                            "urqmd_event_%d" % urqmd_event_id))
                        pid_list = list(set(list(particle_info[:, 0])))
                        for ipid in range(len(pid_list)):
                            particle_name = self.urqmd_pid_dict[pid_list[ipid]]
                            particle_group = urqmd_event_group.create_group(
                                "%s" % particle_name)
                            particle_group.attrs.create(
                                'mass', self.mass_pid_dict[particle_name])
                            idx = particle_info[:, 0] == pid_list[ipid]
                            particle_data = particle_info[idx, 1:9]
                            n_particle = len(particle_data[:, 0])
                            particle_group.attrs.create(
                                'N_particle', n_particle)
                            particle_group.create_dataset(
                                "particle_info", data=particle_data,
                                compression='gzip', compression_opts=9)
                    if urqmd_event_id % 100 == 0:
                        print("processing UrQMD events %d finished."
                              % urqmd_event_id)
                    urqmd_event_id += 1
                    # switch back to header mode
                    data_row_count = 0
                    header_count = 0  # not pointing at the header line yet
                    read_mode = "header_first_part"

        if self.enable_hdf5:
            of_hdf5.flush()
            of_hdf5.close()
        # close connection to commit changes
        if self.enable_sqlite:
            of_db.closeConnection()

    def compute_qn_vectors(self, data_set):
        """
            compute particle spectrum and qn vectors for the given data_set
        """
        n_order = self.flow_analyze_order
        qn_y = np.zeros([n_order, 5]) * (1 + 1j)
        qn_eta = np.zeros([n_order, 5]) * (1 + 1j)

        p_t_min = self.p_t_min
        p_t_max = self.p_t_max
        n_p_t = self.n_p_t
        p_t = np.linspace(p_t_min, p_t_max, n_p_t + 1)
        dp_t = p_t[1] - p_t[0]
        p_t_bin = (p_t[0:-1] + p_t[1:]) / 2.
        qn_p_t_y = np.zeros([n_order, n_p_t]) * (1 + 1j)
        qn_p_t_eta = np.zeros([n_order, n_p_t]) * (1 + 1j)

        p_t_inte_res_y = np.zeros([2 * n_order + 1, 5])
        p_t_inte_res_eta = np.zeros([2 * n_order + 1, 5])
        p_t_diff_res_y = np.zeros([2 * n_order + 1, n_p_t])
        p_t_diff_res_eta = np.zeros([2 * n_order + 1, n_p_t])

        for i in range(len(data_set[:, 0])):
            particle_p_t = data_set[i, 4]
            particle_phi = data_set[i, 5]
            particle_y = data_set[i, 6]
            particle_eta = data_set[i, 7]
            if p_t_min <= particle_p_t < p_t_max:
                # collect pT-integrated variables
                if -2.5 <= particle_y < 2.5:
                    idx_y = int(particle_y + 2.5)
                    for i_order in range(n_order):
                        qn_y[i_order, idx_y] += (
                            np.exp(1j * i_order * particle_phi))
                if -2.5 <= particle_eta < 2.5:
                    idx_eta = int(particle_eta + 2.5)
                    for i_order in range(n_order):
                        qn_eta[i_order, idx_eta] += (
                            np.exp(1j * i_order * particle_phi))

                # collect pT-differential variables
                idx = int((particle_p_t - p_t[0]) / dp_t)
                if -0.5 <= particle_y <= 0.5:
                    for i_order in range(n_order):
                        qn_p_t_y[i_order, idx] += (
                            np.exp(1j * i_order * particle_phi))
                if -0.5 <= particle_eta <= 0.5:
                    for i_order in range(n_order):
                        qn_p_t_eta[i_order, idx] += (
                            np.exp(1j * i_order * particle_phi))

        # normalize qn vectors
        p_t_diff_res_y[0, :] = p_t_bin
        p_t_diff_res_eta[0, :] = p_t_bin
        for i_order in range(1, n_order):
            qn_y[i_order, :] = qn_y[i_order, :] / (qn_y[0, :] + 1e-15)
            qn_eta[i_order, :] = qn_eta[i_order, :] / (qn_eta[0, :] + 1e-15)
            qn_p_t_y[i_order, :] = (
                qn_p_t_y[i_order, :] / (qn_p_t_y[0, :] + 1e-15))
            qn_p_t_eta[i_order, :] = (
                qn_p_t_eta[i_order, :] / (qn_p_t_eta[0, :] + 1e-15))

        for i_order in range(n_order):
            p_t_inte_res_y[2 * i_order + 1:2 * i_order + 3, :] = (
                np.array([np.real(qn_y[i_order, :]),
                          np.imag(qn_y[i_order, :])]))
            p_t_inte_res_eta[2 * i_order + 1:2 * i_order + 3, :] = (
                np.array([np.real(qn_eta[i_order, :]),
                          np.imag(qn_eta[i_order, :])]))
            p_t_diff_res_y[2 * i_order + 1:2 * i_order + 3, :] = (
                np.array([np.real(qn_p_t_y[i_order, :]),
                          np.imag(qn_p_t_y[i_order, :])]))
            p_t_diff_res_eta[2 * i_order + 1:2 * i_order + 3, :] = (
                np.array([np.real(qn_p_t_eta[i_order, :]),
                          np.imag(qn_p_t_eta[i_order, :])]))

        return (p_t_inte_res_y, p_t_inte_res_eta,
                p_t_diff_res_y, p_t_diff_res_eta)

    def collect_particle_info(
            self, folder, subfolder_pattern="event-(\d*)",
            result_filename="particle_list.dat", file_format='UrQMD',
            out_filename="particles",
            particles_to_collect=['charged'], rap_range=(-2.5, 2.5)):
        """
            This function collects particles momentum and space-time
            information from UrQMD outputs into a database
        """
        self.output_filename = out_filename
        # get list of (matched subfolders, event id)
        match_pattern = re.compile(subfolder_pattern)
        matched_subfolders = []
        for folder_index, asubfolder in enumerate(listdir(folder)):
            full_path = path.join(folder, asubfolder)
            # want only folders, not files
            if not path.isdir(full_path):
                continue
            match_result = match_pattern.match(asubfolder)
            # matched!
            if match_result:
                # folder name contains id
                if len(match_result.groups()):
                    hydro_event_id = match_result.groups()[0]
                else:
                    hydro_event_id = folder_index
                matched_subfolders.append((full_path, hydro_event_id))

        rap_type = 'pseudorapidity'

        print("-" * 60)
        print("Collecting particle information from UrQMD outputs...")
        print("-" * 60)
        for asubfolder, hydro_event_id in matched_subfolders:
            print("Collecting %s as with hydro event-id: %s"
                  % (asubfolder, hydro_event_id))
            if file_format == 'UrQMD':
                self.collect_particles_urqmd(
                    asubfolder, hydro_event_id, result_filename,
                    particles_to_collect, rap_type, rap_range)
            elif file_format == 'OSCAR':
                self.collect_particles_oscar(
                    asubfolder, hydro_event_id, result_filename,
                    particles_to_collect, rap_type, rap_range)
            else:
                print("Error: can not recognize the input file format : %s",
                      file_format)
                exit(-1)

    def store_spectra_and_flow(
            self, particle_to_analysis, event_grp, vn_real, vn_imag):
        """
            store the particle spectra and flow data into hdf5 file
        """
        n_order = self.flow_analyze_order
        p_t = np.linspace(self.p_t_min, self.p_t_max, self.n_p_t + 1)
        event_grp.attrs.create('pT', p_t)
        table_names = ['y', 'eta', 'y_pT', 'eta_pT']
        for i_particle in range(len(particle_to_analysis)):
            particle_key = particle_to_analysis[i_particle]
            output_part_grp = event_grp.create_group(particle_key)
            output_part_grp.create_dataset(
                "dNdy", data=np.array(vn_real[0][0][i_particle]),
                compression='gzip')
            output_part_grp.create_dataset(
                "dNdeta", data=np.array(vn_real[0][1][i_particle]),
                compression='gzip')
            output_part_grp.create_dataset(
                "dNdydpT", data=np.array(vn_real[0][2][i_particle]),
                compression='gzip')
            output_part_grp.create_dataset(
                "dNdetadpT", data=np.array(vn_real[0][3][i_particle]),
                compression='gzip')
            for iorder in range(1, n_order):
                for itable in range(len(table_names)):
                    output_part_grp.create_dataset(
                        "v%d_%s_real" % (iorder, table_names[itable]),
                        data=np.array(vn_real[iorder][itable][i_particle]),
                        compression='gzip')
                    output_part_grp.create_dataset(
                        "v%d_%s_imag" % (iorder, table_names[itable]),
                        data=np.array(vn_imag[iorder][itable][i_particle]),
                        compression='gzip')

    def analyze_flow_observables(self, input_filename, output_filename):
        """
            This function performs analysis for single particle spectrum
            and its anisotropy, qn vectors from the hdf5 datafile.
        """
        import h5py

        input_h5 = h5py.File("%s.hdf5" % input_filename, 'r')
        output_h5 = h5py.File("%s.hdf5" % output_filename, 'w')

        events_chunk_size = 10000
        n_order = self.flow_analyze_order

        particle_to_analysis = ['charged', 'pion_p', 'kaon_p', 'proton']
        hydro_event_list = input_h5.keys()
        for i_hydro in range(len(hydro_event_list)):
            print "Analyzing hydro event: %d" % i_hydro
            hydro_key = hydro_event_list[i_hydro]
            urqmd_event_list = input_h5[hydro_key].keys()
            output_hydro_grp = output_h5.create_group(hydro_key)
            for i_urqmd in range(len(urqmd_event_list)):
                if i_urqmd % 100 == 0:
                    print "Analyzing UrQMD events: %d" % i_urqmd
                urqmd_key = urqmd_event_list[i_urqmd]
                particle_list = input_h5[hydro_key][urqmd_key].keys()
                initial = 1
                if i_urqmd % events_chunk_size == 0:
                    grp_name = ("events_%d-%d"
                                % (i_urqmd, i_urqmd + events_chunk_size - 1))
                    output_urqmd_grp = output_hydro_grp.create_group(grp_name)
                    vn_real = []
                    vn_imag = []
                    # vn(y), vn(eta), vn(pT)(y), vn(pT)(eta)
                    for iorder in range(n_order):
                        vn_real.append([[], [], [], []])
                        vn_imag.append([[], [], [], []])
                    initial = 0
                for i_particle in range(len(particle_to_analysis)):
                    particle_key = particle_to_analysis[i_particle]
                    if initial == 0:  # add particle species list
                        for ii in range(4):
                            for iorder in range(n_order):
                                vn_real[iorder][ii].append([])
                                vn_imag[iorder][ii].append([])
                    data = np.array([])
                    if particle_key == 'charged':
                        i_flag = 0
                        for ipart in range(len(self.charged_hadron_list)):
                            part_key = self.charged_hadron_list[ipart]
                            if part_key in particle_list:
                                part_grp = (
                                    input_h5[hydro_key][urqmd_key][part_key])
                                data_temp = part_grp.get('particle_info')
                                if i_flag == 0:
                                    data = data_temp
                                    i_flag = 1
                                else:
                                    data = np.append(data, data_temp, axis=0)
                    else:
                        if particle_key in particle_list:
                            part_grp = (
                                input_h5[hydro_key][urqmd_key][particle_key])
                            data = part_grp.get('particle_info')
                    if len(data) > 0:
                        qn_inte_y, qn_inte_eta, qn_diff_y, qn_diff_eta = (
                            self.compute_qn_vectors(data))
                        for iorder in range(n_order):
                            vn_real[iorder][0][i_particle].append(
                                [i_urqmd] + list(qn_inte_y[2 * iorder + 1, :]))
                            vn_imag[iorder][0][i_particle].append(
                                [i_urqmd] + list(qn_inte_y[2 * iorder + 2, :]))
                            vn_real[iorder][1][i_particle].append(
                                [i_urqmd] + list(
                                    qn_inte_eta[2 * iorder + 1, :]))
                            vn_imag[iorder][1][i_particle].append(
                                [i_urqmd] + list(
                                    qn_inte_eta[2 * iorder + 2, :]))
                            vn_real[iorder][2][i_particle].append(
                                [i_urqmd] + list(qn_diff_y[2 * iorder + 1, :]))
                            vn_imag[iorder][2][i_particle].append(
                                [i_urqmd] + list(qn_diff_y[2 * iorder + 2, :]))
                            vn_real[iorder][3][i_particle].append(
                                [i_urqmd] + list(
                                    qn_diff_eta[2 * iorder + 1, :]))
                            vn_imag[iorder][3][i_particle].append(
                                [i_urqmd] + list(
                                    qn_diff_eta[2 * iorder + 2, :]))
                if i_urqmd % events_chunk_size == events_chunk_size - 1:
                    self.store_spectra_and_flow(
                        particle_to_analysis, output_urqmd_grp,
                        vn_real, vn_imag)
                    vn_real = []
                    vn_imag = []
            if len(vn_real) > 0:
                self.store_spectra_and_flow(
                    particle_to_analysis, output_urqmd_grp, vn_real, vn_imag)
        output_h5.flush()
        input_h5.close()
        output_h5.close()


if __name__ == "__main__":
    try:
        folder = str(argv[1])
    except IndexError:
        print("Usage: particle_filter.py folder")
        exit(0)
    test = ParticleFilter(enable_sqlite=False)
    test.collect_particle_info(folder, file_format='UrQMD')
    test.analyze_flow_observables('particles', 'analyzed')
    #test = ParticleFilter()
    #test.collect_particle_info(folder, result_filename='OSCAR.DAT', 
    #    file_format='OSCAR', out_filename="particles_OSCAR")
    #test.analyze_flow_observables('particles_OSCAR', 'analyzed_OSCAR')
