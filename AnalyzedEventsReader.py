#! /usr/bin/env python

from sys import argv, exit
from os import path
from DBR import SqliteDB
from numpy import *

# define colors
purple = "\033[95m"
green = "\033[92m"
red = "\033[91m"
normal = "\033[0m"


class AnalyzedDataReader(object):
    """
        This class is used to perform event average for all the results
        that are collected from ParticleReader to get final experimental
        observables
    """

    def __init__(self, database):
        """
            Register databases
        """
        # setup database for analysis
        if isinstance(database, str):
            if path.exists(database):
                database = SqliteDB(database)
            else:
                raise ValueError(
                    "AnalyzedDataReader.__init__: input database %s can not "
                    "be found!" % (database))
        if isinstance(database, SqliteDB):
            self.db = database
        else:
            raise TypeError(
                "AnalyzedDataReader.__init__: the input database must be "
                "a string or a SqliteDB database.")

        # setup lookup tables
        self.pid_lookup = dict(self.db.selectFromTable("pid_lookup"))

        # define all charged hadrons
        self.charged_hadron_list = [
            "pion_p", "pion_m", "kaon_p", "kaon_m", "proton", "anti_proton",
            "sigma_p", "sigma_m", "anti_sigma_p", "anti_sigma_m",
            "xi_m", "anti_xi_m"]

        # create index
        if not self.db.doesIndexExist('index_particle_pT_spectra'):
            print("Create index for particle_pT_spectra table ...")
            self.db.executeSQLquery(
                "CREATE INDEX index_particle_pT_spectra ON "
                "particle_pT_spectra(hydro_event_id, urqmd_event_id, pid)")
        for aTable in ['flow_Qn_vectors', 'flow_Qn_vectors_pTdiff']:
            if not self.db.doesIndexExist('index_%s' % aTable):
                print("Create index for table %s ..." % aTable)
                self.db.executeSQLquery("CREATE INDEX index_%s ON "
                                        "%s(hydro_event_id, urqmd_event_id, "
                                        "pid, weight_type, n)"
                                        % (aTable, aTable))

        # get number of events
        # pre-collect it to shorten the initial loading time of the database
        if self.db.createTableIfNotExists(
                "number_of_events", (("Nev_tot", "integer"),
                                     ("Nev_hydro", "integer"))):
            self.tot_nev = self.get_number_of_total_events()
            self.hydro_nev = self.get_number_of_hydro_events()
            self.db.insertIntoTable("number_of_events",
                                    (int(self.tot_nev), int(self.hydro_nev)))
            self.db._dbCon.commit()  # commit changes
        else:
            self.tot_nev = self.db.executeSQLquery(
                "select Nev_tot from number_of_events").fetchall()[0][0]
            self.hydro_nev = self.db.executeSQLquery(
                "select Nev_hydro from number_of_events").fetchall()[0][0]

        temp_flag = self.db.createTableIfNotExists(
            "UrQMD_NevList", (("hydroEventId", "integer"),
                              ("Number_of_UrQMDevents", "integer")))
        urqmd_nev_array = zeros(self.hydro_nev)
        for hydroEventId in range(1, self.hydro_nev + 1):
            urqmd_nev = self.get_number_of_urqmd_events(hydroEventId)
            urqmd_nev_array[hydroEventId - 1] = urqmd_nev
            if temp_flag:
                self.db.insertIntoTable("UrQMD_NevList",
                                        (int(hydroEventId), int(urqmd_nev)))
        self.db._dbCon.commit()  # commit changes

        self.process_nev = 10000
        # determine retrieve event boundaries
        self.nev_bin = int(self.tot_nev / self.process_nev) + 2
        if self.tot_nev % self.process_nev == 0: self.nev_bin -= 1
        self.event_bound_hydro = ones(self.nev_bin)
        self.event_bound_urqmd = ones(self.nev_bin)
        ihydro_ev = 1
        ibin = 1
        temp = urqmd_nev_array[0]
        while ihydro_ev <= self.hydro_nev:
            if temp > ibin * self.process_nev:
                self.event_bound_hydro[ibin] = ihydro_ev
                ibin += 1
            else:
                ihydro_ev += 1
                if ihydro_ev < self.hydro_nev:
                    temp += urqmd_nev_array[ihydro_ev - 1]
                else:
                    self.event_bound_hydro[ibin] = ihydro_ev
        for ibin in range(1, self.nev_bin):
            self.event_bound_urqmd[ibin] = (
                ibin * self.process_nev
                - sum(urqmd_nev_array[0:self.event_bound_hydro[ibin] - 1]))

    ###########################################################################
    # functions to get number of events
    ########################################################################### 
    def get_number_of_hydro_events(self):
        """
            return total number hydro events stored in the database
        """
        nev = self.db.executeSQLquery(
            "select count(*) from (select distinct hydro_event_id "
            "from particle_pT_spectra)").fetchall()[0][0]
        return nev

    def get_number_of_urqmd_events(self, hydroeventid):
        """
            return number of UrQMD events for the given hydro event
        """
        nev = self.db.executeSQLquery(
            "select count(*) from (select distinct urqmd_event_id from "
            "particle_pT_spectra where hydro_event_id = %d)" % hydroeventid
        ).fetchall()[0][0]
        return nev

    def get_number_of_total_events(self):
        """
            return total number of events stored in the database
        """
        hydro_event_id = self.db.executeSQLquery(
            "select distinct hydro_event_id from particle_pT_spectra"
        ).fetchall()
        nev = 0
        for iev in range(len(hydro_event_id)):
            nev += self.get_number_of_urqmd_events(hydro_event_id[iev][0])
        return nev

    def get_pid_string(self, particleName):
        if isinstance(particleName, list):
            pidList = []
            for aPart in particleName:
                pidList.append(str(self.pid_lookup[aPart]))
            pidstring = " or ".join(map(lambda (x): 'pid = ' + x, pidList))
        else:
            pid = self.pid_lookup[particleName]
            if pid == 1:  # all charged hadrons
                pidstring = self.get_pid_string(self.charged_hadron_list)
            else:
                pidstring = "pid = %d" % pid
        return (pidstring)

    ###########################################################################
    # functions to collect particle spectra and yields
    ########################################################################### 
    def get_particle_spectra(self, particle_name, pT_range=linspace(0, 3, 31),
                             rap_type='rapidity'):
        """
            This function performs event average for particle pT-spectra
            from the database and interpolates to the desired pT values
            specified by the users.
            It returns (pT, dN/(dydpT), dN/(dydpT)_err)
        """
        print("processing particle spectra for %s ... " % particle_name)
        eps = 1e-15
        pid = self.pid_lookup[particle_name]
        if rap_type == 'rapidity':
            analyzed_table_name = 'particle_pT_spectra'
        elif rap_type == 'pseudorapidity':
            analyzed_table_name = 'particle_pT_spectra_eta'
        else:
            raise ValueError("unrecognized rap_type: %s" % rap_type)

        npT = len(self.db.executeSQLquery(
            "select pT from %s where hydro_event_id = %d and "
            "urqmd_event_id = %d and pid = %d"
            % (analyzed_table_name, 1, 1, pid)).fetchall())

        dN_avg = zeros([npT, 3])
        #fetch data
        for ibin in range(1, self.nev_bin):
            print("processing events %d to %d ..."
                  % ((ibin - 1) * self.process_nev, ibin * self.process_nev))
            hydro_ev_bound_low = self.event_bound_hydro[ibin - 1]
            hydro_ev_bound_high = self.event_bound_hydro[ibin]
            urqmd_ev_bound_low = self.event_bound_urqmd[ibin - 1]
            urqmd_ev_bound_high = self.event_bound_urqmd[ibin]
            if hydro_ev_bound_low == hydro_ev_bound_high:
                temp_data = array(self.db.executeSQLquery(
                    "select pT, dN_dydpT from %s where pid = %d and "
                    "hydro_event_id = %d and (%d <= urqmd_event_id "
                    "and urqmd_event_id < %d)"
                    % (analyzed_table_name, pid, hydro_ev_bound_low,
                       urqmd_ev_bound_low, urqmd_ev_bound_high)).fetchall())
            else:
                temp_data = array(self.db.executeSQLquery(
                    "select pT, dN_dydpT from %s where pid = %d and "
                    "((hydro_event_id = %d and urqmd_event_id >= %d) "
                    " or (%d < hydro_event_id and hydro_event_id < %d) "
                    " or (hydro_event_id = %d and urqmd_event_id < %d))"
                    % (analyzed_table_name, pid, hydro_ev_bound_low,
                       urqmd_ev_bound_low, hydro_ev_bound_low,
                       hydro_ev_bound_high, hydro_ev_bound_high,
                       urqmd_ev_bound_high)).fetchall())
            for ipT in range(npT):
                dN_avg[ipT, 0] += (
                    sum(temp_data[ipT::npT, 0] * temp_data[ipT::npT, 1]))
                dN_avg[ipT, 1] += sum(temp_data[ipT::npT, 1])
                dN_avg[ipT, 2] += sum(temp_data[ipT::npT, 1] ** 2)

        # calculate mean pT, <dN/dydpT>, and <dN/dydpT>_err 
        dN_avg[:, 0] = dN_avg[:, 0] / dN_avg[:, 1]
        dN_avg[:, 1] = dN_avg[:, 1] / self.tot_nev
        dN_avg[:, 2] = (sqrt(dN_avg[:, 2] / self.tot_nev - dN_avg[:, 1] ** 2)
                        / sqrt(self.tot_nev - 1))

        #interpolate results to desired pT range
        dNdyinterp = exp(
            interp(pT_range, dN_avg[:, 0], log(dN_avg[:, 1] + eps)))
        dNdyinterp_err = exp(
            interp(pT_range, dN_avg[:, 0], log(dN_avg[:, 2] + eps)))
        results = array([pT_range, dNdyinterp, dNdyinterp_err])
        return transpose(results)

    def get_particle_yield_vs_rap(self, particle_name, rap_type='rapidity',
                                  rap_range=linspace(-2.0, 2.0, 41)):
        """
            It returns event averaged particle yield vs rapidity
            or pseudorapidity. The range is specified by the user.
            It returns (rap, dN/(drap), dN/(drap)_err)
        """
        print("get %s dependence of particle yield for %s"
              % (rap_type, particle_name))
        pid = self.pid_lookup[particle_name]
        if rap_type == 'rapidity':
            analyzed_table_name = 'particle_yield_vs_rap'
        elif rap_type == 'pseudorapidity':
            analyzed_table_name = 'particle_yield_vs_psedurap'
        else:
            raise ValueError("unrecognized rap_type: %s" % rap_type)

        nrap = len(self.db.executeSQLquery(
            "select rap from %s where hydro_event_id = %d and "
            "urqmd_event_id = %d and pid = %d"
            % (analyzed_table_name, 1, 1, pid)).fetchall())

        dN_avg = zeros([nrap, 3])
        #fetch data
        for ibin in range(1, self.nev_bin):
            print("processing events %d to %d ..."
                  % ((ibin - 1) * self.process_nev, ibin * self.process_nev))
            hydro_ev_bound_low = self.event_bound_hydro[ibin - 1]
            hydro_ev_bound_high = self.event_bound_hydro[ibin]
            urqmd_ev_bound_low = self.event_bound_urqmd[ibin - 1]
            urqmd_ev_bound_high = self.event_bound_urqmd[ibin]
            if hydro_ev_bound_low == hydro_ev_bound_high:
                temp_data = array(self.db.executeSQLquery(
                    "select rap, dN_drap from %s where pid = %d and "
                    "hydro_event_id = %d and (%d <= urqmd_event_id "
                    "and urqmd_event_id < %d)"
                    % (analyzed_table_name, pid, hydro_ev_bound_low,
                       urqmd_ev_bound_low, urqmd_ev_bound_high)).fetchall())
            else:
                temp_data = array(self.db.executeSQLquery(
                    "select rap, dN_drap from %s where pid = %d and "
                    "((hydro_event_id = %d and urqmd_event_id >= %d) "
                    " or (%d < hydro_event_id and hydro_event_id < %d) "
                    " or (hydro_event_id = %d and urqmd_event_id < %d))"
                    % (analyzed_table_name, pid, hydro_ev_bound_low,
                       urqmd_ev_bound_low, hydro_ev_bound_low,
                       hydro_ev_bound_high, hydro_ev_bound_high,
                       urqmd_ev_bound_high)).fetchall())
            for irap in range(nrap):
                dN_avg[irap, 0] += sum(
                    temp_data[irap::nrap, 0] * temp_data[irap::nrap, 1])
                dN_avg[irap, 1] += sum(temp_data[irap::nrap, 1])
                dN_avg[irap, 2] += sum(temp_data[irap::nrap, 1] ** 2)

        # calculate mean rap, <dN/drap>, and <dN/drap>_err 
        dN_avg[:, 0] = dN_avg[:, 0] / dN_avg[:, 1]
        dN_avg[:, 1] = dN_avg[:, 1] / self.tot_nev
        dN_avg[:, 2] = (sqrt(dN_avg[:, 2] / self.tot_nev - dN_avg[:, 1] ** 2)
                        / sqrt(self.tot_nev - 1))

        #interpolate results to desired rap range
        dNdyinterp = interp(rap_range, dN_avg[:, 0], dN_avg[:, 1])
        dNdyinterp_err = interp(rap_range, dN_avg[:, 0], dN_avg[:, 2])
        results = array([rap_range, dNdyinterp, dNdyinterp_err])
        return transpose(results)

    def get_particle_yield(self, particle_name, rap_type='rapidity',
                           rap_range=(-0.5, 0.5)):
        """
            return the integrated particle yield N of particle species 
            "particle_name" within the given rapidity or pseudorapidity 
            range by users
        """
        print("get particle yield for %s" % particle_name)
        pid = self.pid_lookup[particle_name]
        if rap_type == 'rapidity':
            analyzed_table_name = 'particle_yield_vs_rap'
        elif rap_type == 'pseudorapidity':
            analyzed_table_name = 'particle_yield_vs_psedurap'
        else:
            raise ValueError("unrecognized rap_type: %s" % rap_type)

        nrap = len(self.db.executeSQLquery(
            "select rap from %s where hydro_event_id = %d and "
            "urqmd_event_id = %d and pid = %d"
            % (analyzed_table_name, 1, 1, pid)).fetchall())
        drap = 0.1

        mean_rap = 0.0
        particle_yield = 0.0
        particle_yield_err = 0.0

        #fetch data
        for ibin in range(1, self.nev_bin):
            print("processing events %d to %d ..."
                  % ((ibin - 1) * self.process_nev, ibin * self.process_nev))
            hydro_ev_bound_low = self.event_bound_hydro[ibin - 1]
            hydro_ev_bound_high = self.event_bound_hydro[ibin]
            urqmd_ev_bound_low = self.event_bound_urqmd[ibin - 1]
            urqmd_ev_bound_high = self.event_bound_urqmd[ibin]
            if hydro_ev_bound_low == hydro_ev_bound_high:
                temp_data = array(self.db.executeSQLquery(
                    "select rap, dN_drap from %s where pid = %d and "
                    "hydro_event_id = %d and (%d <= urqmd_event_id "
                    "and urqmd_event_id < %d)"
                    % (analyzed_table_name, pid, hydro_ev_bound_low,
                       urqmd_ev_bound_low, urqmd_ev_bound_high)).fetchall())
            else:
                temp_data = array(self.db.executeSQLquery(
                    "select rap, dN_drap from %s where pid = %d and "
                    "((hydro_event_id = %d and urqmd_event_id >= %d) "
                    " or (%d < hydro_event_id and hydro_event_id < %d) "
                    " or (hydro_event_id = %d and urqmd_event_id < %d))"
                    % (analyzed_table_name, pid, hydro_ev_bound_low,
                       urqmd_ev_bound_low, hydro_ev_bound_low,
                       hydro_ev_bound_high, hydro_ev_bound_high,
                       urqmd_ev_bound_high)).fetchall())
            cut_data = temp_data[temp_data[:, 0] > rap_range[0], :]
            cut_data2 = cut_data[cut_data[:, 0] < rap_range[1], :]
            mean_rap += sum(cut_data2[:, 0] * cut_data2[:, 1])
            particle_yield += sum(cut_data2[:, 1])
            particle_yield_err += sum(cut_data2[:, 1] ** 2)

        # calculate mean rap, <dN>, and <dN>_err 
        mean_rap = mean_rap / particle_yield
        particle_yield = particle_yield / self.tot_nev * drap
        particle_yield_err = (
            (sqrt(particle_yield_err / self.tot_nev - particle_yield ** 2)
             / sqrt(self.tot_nev - 1)) * drap)
        return (mean_rap, particle_yield, particle_yield_err)

    ###########################################################################
    # functions to collect particle emission function
    ########################################################################### 
    def get_particle_yield_vs_spatial_variable(
            self, particle_name, sv_type, sv_range, rap_type):
        """
            This function performs event average for particle yield per
            spatial variable from the database and interpolates to the 
            desired sv values specified by the users.
            It returns (sv, dN/(dsv), dN/(dsv)_err)
        """
        print("get dN/d%s for %s" % (sv_type, particle_name))
        eps = 1e-15
        pid = self.pid_lookup[particle_name]
        if rap_type == 'rapidity':
            analyzed_table_name = 'particle_emission_d%s' % sv_type
        elif rap_type == 'pseudorapidity':
            analyzed_table_name = 'particle_emission_d%s_eta' % sv_type
        else:
            raise ValueError("AnalyzedDataReader.get_particle_yield_vs_"
                             "spatial_variable: invalid input rap_type : %s"
                             % rap_type)

        nsv = len(self.db.executeSQLquery(
            "select %s from %s where hydro_event_id = %d and "
            "urqmd_event_id = %d and pid = %d"
            % (sv_type, analyzed_table_name, 1, 1, pid)).fetchall())

        dN_avg = zeros([nsv, 3])

        #fetch data
        for ibin in range(1, self.nev_bin):
            print("processing events %d to %d ..."
                  % ((ibin - 1) * self.process_nev, ibin * self.process_nev))
            hydro_ev_bound_low = self.event_bound_hydro[ibin - 1]
            hydro_ev_bound_high = self.event_bound_hydro[ibin]
            urqmd_ev_bound_low = self.event_bound_urqmd[ibin - 1]
            urqmd_ev_bound_high = self.event_bound_urqmd[ibin]
            if hydro_ev_bound_low == hydro_ev_bound_high:
                temp_data = array(self.db.executeSQLquery(
                    "select %s, dN_d%s from %s where pid = %d and "
                    "hydro_event_id = %d and (%d <= urqmd_event_id "
                    "and urqmd_event_id < %d)"
                    % (sv_type, sv_type, analyzed_table_name, pid,
                       hydro_ev_bound_low, urqmd_ev_bound_low,
                       urqmd_ev_bound_high)
                ).fetchall())
            else:
                temp_data = array(self.db.executeSQLquery(
                    "select %s, dN_d%s from %s where pid = %d and "
                    "((hydro_event_id = %d and urqmd_event_id >= %d) "
                    " or (%d < hydro_event_id and hydro_event_id < %d) "
                    " or (hydro_event_id = %d and urqmd_event_id < %d))"
                    % (sv_type, sv_type, analyzed_table_name, pid,
                       hydro_ev_bound_low, urqmd_ev_bound_low,
                       hydro_ev_bound_low, hydro_ev_bound_high,
                       hydro_ev_bound_high, urqmd_ev_bound_high)
                ).fetchall())
            for isv in range(nsv):
                dN_avg[isv, 0] += sum(
                    temp_data[isv::nsv, 0] * temp_data[isv::nsv, 1])
                dN_avg[isv, 1] += sum(temp_data[isv::nsv, 1])
                dN_avg[isv, 2] += sum(temp_data[isv::nsv, 1] ** 2)

        # calculate <sv>, <dN/dsv>, and <dN/dsv>_err 
        dN_avg[:, 0] = dN_avg[:, 0] / (dN_avg[:, 1] + eps)
        dN_avg[:, 1] = dN_avg[:, 1] / self.tot_nev
        dN_avg[:, 2] = (sqrt(dN_avg[:, 2] / self.tot_nev - dN_avg[:, 1] ** 2)
                        / sqrt(self.tot_nev - 1))

        # delete zero components for interpolation
        idx = []
        for i in range(len(dN_avg[:, 0])):
            if abs(dN_avg[i, 0]) < 1e-15:
                idx.append(i)
        sv_mean = delete(dN_avg[:, 0], idx)
        dN_dsv = delete(dN_avg[:, 1], idx)
        dN_dsv_err = delete(dN_avg[:, 2], idx)

        #interpolate results to desired sv range
        dNdyinterp = interp(sv_range, sv_mean, dN_dsv)
        dNdyinterp_err = interp(sv_range, sv_mean, dN_dsv_err)
        results = array([sv_range, dNdyinterp, dNdyinterp_err])
        return transpose(results)

    ###########################################################################
    # functions to collect particle anisotropic flows
    ###########################################################################
    def get_all_qn_vectors(self, particle_name, order):
        """
        This function return p_T-integrated Qn vectors for all events
        :param particle_name: particle species
        :param order: harmonic order
        """
        analyzed_table_name_inte = 'flow_Qn_vectors'
        pid = self.pid_lookup[particle_name]
        temp_data = array(self.db.executeSQLquery(
            "select Nparticle, Qn_real, Qn_imag from %s "
            "where pid = %d and weight_type = '1' and n = %d "
            % (analyzed_table_name_inte, pid, order)
        ).fetchall())
        return temp_data

    def get_all_qn_vectors_pTdiff(self, particle_name, order):
        """
        This function return p_T-differential Qn vectors for all events
        :param particle_name: particle species
        :param order: harmonic order
        """
        analyzed_table_name_diff = 'flow_Qn_vectors_pTdiff'
        pid = self.pid_lookup[particle_name]
        npT = len(self.db.executeSQLquery(
            "select pT from %s where hydro_event_id = %d and "
            "urqmd_event_id = %d and pid = %d and weight_type = '1' and n = %d"
            % (analyzed_table_name_diff, 1, 1, pid, order)).fetchall())
        temp_data = array(self.db.executeSQLquery(
            "select pT, Nparticle, Qn_real, Qn_imag from %s "
            "where pid = %d and weight_type = '1' and n = %d "
            % (analyzed_table_name_diff, pid, order)
        ).fetchall())
        return npT, temp_data

    def get_avg_diffvn_flow(
            self, particle_name, order, psi_r=0.,
            pT_range=linspace(0.0, 3.0, 31)):
        """
            compute <cos(n*(phi_i - psiR))>_ev over all events and interpolate
            the results to desired pT points given by the user
        """
        print("collect averaged diff vn flow of %s ..." % particle_name)
        eps = 1e-15
        pid = self.pid_lookup[particle_name]
        analyzed_table_name = 'flow_Qn_vectors_pTdiff'
        npT = len(self.db.executeSQLquery(
            "select pT from %s where hydro_event_id = %d and "
            "urqmd_event_id = %d and pid = %d and weight_type = '1' and n = %d"
            % (analyzed_table_name, 1, 1, pid, order)).fetchall())

        vn_avg = zeros([npT, 5])
        vn_real = zeros(npT)
        vn_imag = zeros(npT)
        vn_real_err = zeros(npT)
        vn_imag_err = zeros(npT)
        totalN = zeros(npT)
        nev_pT = zeros(npT)

        cos_psi_r = cos(order * psi_r)
        sin_psi_r = sin(order * psi_r)

        #fetch data
        for ibin in range(1, self.nev_bin):
            print("processing events %d to %d ..."
                  % ((ibin - 1) * self.process_nev, ibin * self.process_nev))
            hydro_ev_bound_low = self.event_bound_hydro[ibin - 1]
            hydro_ev_bound_high = self.event_bound_hydro[ibin]
            urqmd_ev_bound_low = self.event_bound_urqmd[ibin - 1]
            urqmd_ev_bound_high = self.event_bound_urqmd[ibin]
            if hydro_ev_bound_low == hydro_ev_bound_high:
                temp_data = array(self.db.executeSQLquery(
                    "select pT, Nparticle, Qn_real, Qn_imag from %s "
                    "where pid = %d and weight_type = '1' and n = %d and "
                    "hydro_event_id = %d and "
                    "(%d <= urqmd_event_id and urqmd_event_id < %d)"
                    % (analyzed_table_name, pid, order, hydro_ev_bound_low,
                       urqmd_ev_bound_low, urqmd_ev_bound_high)
                ).fetchall())
            else:
                temp_data = array(self.db.executeSQLquery(
                    "select pT, Nparticle, Qn_real, Qn_imag from %s "
                    "where pid = %d and weight_type = '1' and n = %d and "
                    "((hydro_event_id = %d and urqmd_event_id >= %d) "
                    " or (%d < hydro_event_id and hydro_event_id < %d) "
                    " or (hydro_event_id = %d and urqmd_event_id < %d))"
                    % (analyzed_table_name, pid, order,
                       hydro_ev_bound_low, urqmd_ev_bound_low,
                       hydro_ev_bound_low, hydro_ev_bound_high,
                       hydro_ev_bound_high, urqmd_ev_bound_high)
                ).fetchall())
            for ipT in range(npT):
                single_pT_data = temp_data[ipT::npT, :]
                vn_avg[ipT, 0] += sum(
                    single_pT_data[:, 0] * single_pT_data[:, 1])
                totalN[ipT] += sum(single_pT_data[:, 1])

                nev_pT[ipT] += len(single_pT_data[single_pT_data[:, 1] > 0, 1])
                temp_real = (single_pT_data[:, 2] * cos_psi_r
                             + single_pT_data[:, 3] * sin_psi_r)
                temp_imag = (single_pT_data[:, 3] * cos_psi_r
                             - single_pT_data[:, 2] * sin_psi_r)
                vn_real[ipT] += sum(temp_real)
                vn_imag[ipT] += sum(temp_imag)
                vn_real_err[ipT] += sum(temp_real ** 2.)
                vn_imag_err[ipT] += sum(temp_imag ** 2.)
        vn_avg[:, 0] = vn_avg[:, 0] / totalN
        vn_real = vn_real / nev_pT
        vn_imag = vn_imag / nev_pT
        vn_real_err = sqrt(vn_real_err / nev_pT - vn_real ** 2) / sqrt(
            nev_pT - 1)
        vn_imag_err = sqrt(vn_imag_err / nev_pT - vn_imag ** 2) / sqrt(
            nev_pT - 1)
        vn_avg[:, 1] = vn_real
        vn_avg[:, 2] = vn_real_err
        vn_avg[:, 3] = vn_imag
        vn_avg[:, 4] = vn_imag_err

        #interpolate results to desired pT range
        vn_real_interp = interp(pT_range, vn_avg[:, 0], vn_avg[:, 1])
        vn_real_interp_err = interp(pT_range, vn_avg[:, 0], vn_avg[:, 2])
        vn_imag_interp = interp(pT_range, vn_avg[:, 0], vn_avg[:, 3])
        vn_imag_interp_err = interp(pT_range, vn_avg[:, 0], vn_avg[:, 4])
        results = array([pT_range, vn_real_interp, vn_real_interp_err,
                         vn_imag_interp, vn_imag_interp_err])
        return transpose(results)

    def get_avg_intevn_flow(
            self, particle_name, order, psi_r=0., pT_range=(0.0, 3.0)):
        """
            compute pT integrated <cos(n*(phi_i - psiR))>_ev averaged over 
            all events the pT integrated range is given by the user
        """
        print("collect averaged inte vn flow of %s ..." % particle_name)
        pid = self.pid_lookup[particle_name]
        analyzed_table_name = 'flow_Qn_vectors_pTdiff'

        vn_avg = zeros(5)
        vn_real = 0.0
        vn_imag = 0.0
        vn_real_err = 0.0
        vn_imag_err = 0.0
        totalN = 0
        nev = 0

        cos_psi_r = cos(order * psi_r)
        sin_psi_r = sin(order * psi_r)

        npT = len(self.db.executeSQLquery(
            "select pT from %s where hydro_event_id = %d and "
            "urqmd_event_id = %d and pid = %d and weight_type = '1' and "
            "n = %d and (%g <= pT and pT <= %g)"
            % (analyzed_table_name, 1, 1, pid, order, pT_range[0],
               pT_range[1])
        ).fetchall())

        #fetch data
        for ibin in range(1, self.nev_bin):
            print("processing events %d to %d ..."
                  % ((ibin - 1) * self.process_nev, ibin * self.process_nev))
            hydro_ev_bound_low = self.event_bound_hydro[ibin - 1]
            hydro_ev_bound_high = self.event_bound_hydro[ibin]
            urqmd_ev_bound_low = self.event_bound_urqmd[ibin - 1]
            urqmd_ev_bound_high = self.event_bound_urqmd[ibin]
            if hydro_ev_bound_low == hydro_ev_bound_high:
                temp_data = array(self.db.executeSQLquery(
                    "select pT, Nparticle, Qn_real, Qn_imag from %s "
                    "where pid = %d and weight_type = '1' and n = %d and "
                    "hydro_event_id = %d and "
                    "(%d <= urqmd_event_id and urqmd_event_id < %d) and "
                    "(%g <= pT and pT <= %g)"
                    % (analyzed_table_name, pid, order, hydro_ev_bound_low,
                       urqmd_ev_bound_low, urqmd_ev_bound_high,
                       pT_range[0], pT_range[1])
                ).fetchall())
            else:
                temp_data = array(self.db.executeSQLquery(
                    "select pT, Nparticle, Qn_real, Qn_imag from %s "
                    "where pid = %d and weight_type = '1' and n = %d and "
                    "((hydro_event_id = %d and urqmd_event_id >= %d) "
                    " or (%d < hydro_event_id and hydro_event_id < %d) "
                    " or (hydro_event_id = %d and urqmd_event_id < %d)) and "
                    "(%g <= pT and pT <= %g)"
                    % (analyzed_table_name, pid, order,
                       hydro_ev_bound_low, urqmd_ev_bound_low,
                       hydro_ev_bound_low, hydro_ev_bound_high,
                       hydro_ev_bound_high, urqmd_ev_bound_high,
                       pT_range[0], pT_range[1])
                ).fetchall())
            vn_avg[0] += sum(temp_data[:, 0] * temp_data[:, 1])  #<pT>
            totalN += sum(temp_data[:, 1])
            temp_nev = int(len(temp_data[:, 0]) / npT)
            for iev in range(temp_nev):
                ev_data = temp_data[iev * npT:(iev + 1) * npT, :]
                nparticle = sum(ev_data[:, 1])
                if nparticle == 0: continue
                nev += 1
                # sum of pT bins to construct for pT integrated Qn
                pTinte_Qn_x = sum(ev_data[:, 1] * ev_data[:, 2]) / nparticle
                pTinte_Qn_y = sum(ev_data[:, 1] * ev_data[:, 3]) / nparticle
                temp_real = pTinte_Qn_x * cos_psi_r + pTinte_Qn_y * sin_psi_r
                temp_imag = pTinte_Qn_y * cos_psi_r - pTinte_Qn_x * sin_psi_r
                vn_real += temp_real
                vn_imag += temp_imag
                vn_real_err += temp_real ** 2.
                vn_imag_err += temp_imag ** 2.
        vn_avg[0] = vn_avg[0] / totalN
        vn_real = vn_real / nev
        vn_imag = vn_imag / nev
        vn_real_err = sqrt(vn_real_err / nev - vn_real ** 2) / sqrt(nev - 1)
        vn_imag_err = sqrt(vn_imag_err / nev - vn_imag ** 2) / sqrt(nev - 1)
        vn_avg[1] = vn_real
        vn_avg[2] = vn_real_err
        vn_avg[3] = vn_imag
        vn_avg[4] = vn_imag_err

        return vn_avg

    def get_diffvn_flow(
            self, particle_name, method, order,
            pT_range=linspace(0.0, 3.0, 31)):
        """
            compute pT differential vn(pT) according event plane or scalar 
            product method. The results are interpolated to desired pT points 
            given by the user.

            event plane method:
                <Qn*QnA/|QnA|>_ev/sqrt(<QnA/|QnA|>_ev*<QnB/|QnB|>_ev)
            scalar product method:
                <Qn*QnA>_ev/sqrt(<QnA>_ev*<QnB>_ev)

            QnA and QnB are taken as pT integrated Qn vectors from two 
            independent sub events
            return (pT, vn_real, vn_real_err, vn_imag, vn_imag_err)
        """
        if method == 'event_plane':
            print("collect pT differential vn{EP}(pT) of %s ..."
                  % particle_name)
        elif method == 'scalar_product':
            print("collect pT differential vn{SP}(pT) of %s ..."
                  % particle_name)
        else:
            raise ValueError("AnalyzedDataReader.get_diffvn_flow: "
                             "invalid method : %s" % method)

        eps = 1e-15
        pid = self.pid_lookup[particle_name]
        analyzed_table_name_diff = 'flow_Qn_vectors_pTdiff'
        analyzed_table_name_inte = 'flow_Qn_vectors'

        npT = len(self.db.executeSQLquery(
            "select pT from %s where hydro_event_id = %d and "
            "urqmd_event_id = %d and pid = %d and weight_type = '1' and n = %d"
            % (analyzed_table_name_diff, 1, 1, pid, order)).fetchall())

        vn_avg = zeros([npT, 5])
        vn_real = zeros(npT)
        vn_imag = zeros(npT)
        vn_real_err = zeros(npT)
        vn_imag_err = zeros(npT)
        totalN = zeros(npT)
        nev_pT = zeros(npT)

        resolutionFactor = 0.0
        nev_resolution = 0
        #fetch data
        for ibin in range(1, self.nev_bin):
            print("processing events %d to %d ..."
                  % ((ibin - 1) * self.process_nev, ibin * self.process_nev))
            hydro_ev_bound_low = self.event_bound_hydro[ibin - 1]
            hydro_ev_bound_high = self.event_bound_hydro[ibin]
            urqmd_ev_bound_low = self.event_bound_urqmd[ibin - 1]
            urqmd_ev_bound_high = self.event_bound_urqmd[ibin]
            if hydro_ev_bound_low == hydro_ev_bound_high:
                temp_data = array(self.db.executeSQLquery(
                    "select pT, Nparticle, Qn_real, Qn_imag from %s "
                    "where pid = %d and weight_type = '1' and n = %d and "
                    "hydro_event_id = %d and "
                    "(%d <= urqmd_event_id and urqmd_event_id < %d)"
                    % (analyzed_table_name_diff, pid, order,
                       hydro_ev_bound_low, urqmd_ev_bound_low,
                       urqmd_ev_bound_high)
                ).fetchall())
                ref_data = array(self.db.executeSQLquery(
                    "select QnA_real, QnA_imag, QnB_real, QnB_imag, "
                    "QnC_real, QnC_imag, QnD_real, QnD_imag from %s "
                    "where pid = 1 and weight_type = '1' and n = %d and "
                    "hydro_event_id = %d and "
                    "(%d <= urqmd_event_id and urqmd_event_id < %d)"
                    % (analyzed_table_name_inte, order, hydro_ev_bound_low,
                       urqmd_ev_bound_low, urqmd_ev_bound_high)
                ).fetchall())
            else:
                temp_data = array(self.db.executeSQLquery(
                    "select pT, Nparticle, Qn_real, Qn_imag from %s "
                    "where pid = %d and weight_type = '1' and n = %d and "
                    "((hydro_event_id = %d and urqmd_event_id >= %d) "
                    " or (%d < hydro_event_id and hydro_event_id < %d) "
                    " or (hydro_event_id = %d and urqmd_event_id < %d))"
                    % (analyzed_table_name_diff, pid, order,
                       hydro_ev_bound_low, urqmd_ev_bound_low,
                       hydro_ev_bound_low, hydro_ev_bound_high,
                       hydro_ev_bound_high, urqmd_ev_bound_high)
                ).fetchall())
                ref_data = array(self.db.executeSQLquery(
                    "select QnA_real, QnA_imag, QnB_real, QnB_imag, "
                    "QnC_real, QnC_imag, QnD_real, QnD_imag from %s "
                    "where pid = 1 and weight_type = '1' and n = %d and "
                    "((hydro_event_id = %d and urqmd_event_id >= %d) "
                    " or (%d < hydro_event_id and hydro_event_id < %d) "
                    " or (hydro_event_id = %d and urqmd_event_id < %d))"
                    % (analyzed_table_name_inte, order, hydro_ev_bound_low,
                       urqmd_ev_bound_low, hydro_ev_bound_low,
                       hydro_ev_bound_high, hydro_ev_bound_high,
                       urqmd_ev_bound_high)
                ).fetchall())
            ref_Qn_AC_real = 0.5 * (ref_data[:, 0] + ref_data[:, 4])
            ref_Qn_AC_imag = 0.5 * (ref_data[:, 1] + ref_data[:, 5])
            ref_Qn_BD_real = 0.5 * (ref_data[:, 2] + ref_data[:, 6])
            ref_Qn_BD_imag = 0.5 * (ref_data[:, 3] + ref_data[:, 7])
            ref_QnAC = sqrt(ref_Qn_AC_real ** 2. + ref_Qn_AC_imag ** 2)
            ref_QnBD = sqrt(ref_Qn_BD_real ** 2. + ref_Qn_BD_imag ** 2)
            if method == 'event_plane':
                resolutionFactor += sum((ref_Qn_AC_real * ref_Qn_BD_real
                                         + ref_Qn_AC_imag * ref_Qn_BD_imag)
                                        / ref_QnAC / ref_QnBD)
            else:
                resolutionFactor += sum(ref_Qn_AC_real * ref_Qn_BD_real
                                        + ref_Qn_AC_imag * ref_Qn_BD_imag)
            nev_resolution += len(ref_data[:, 0])
            for ipT in range(npT):
                single_pT_data = temp_data[ipT::npT, :]
                vn_avg[ipT, 0] += sum(
                    single_pT_data[:, 0] * single_pT_data[:, 1])
                totalN[ipT] += sum(single_pT_data[:, 1])

                nev_pT[ipT] += len(single_pT_data[single_pT_data[:, 1] > 0, 1])

                if method == 'event_plane':
                    temp_real = (
                        (single_pT_data[:, 2] * ref_Qn_AC_real
                         + single_pT_data[:, 3] * ref_Qn_AC_imag) / ref_QnAC)
                    temp_imag = (
                        (single_pT_data[:, 3] * ref_Qn_AC_real
                         + single_pT_data[:, 2] * ref_Qn_AC_imag) / ref_QnAC)
                else:
                    temp_real = (single_pT_data[:, 2] * ref_Qn_AC_real
                                 + single_pT_data[:, 3] * ref_Qn_AC_imag)
                    temp_imag = (single_pT_data[:, 3] * ref_Qn_AC_real
                                 + single_pT_data[:, 2] * ref_Qn_AC_imag)

                vn_real[ipT] += sum(temp_real)
                vn_imag[ipT] += sum(temp_imag)
                vn_real_err[ipT] += sum(temp_real ** 2.)
                vn_imag_err[ipT] += sum(temp_imag ** 2.)

        resolutionFactor = sqrt(resolutionFactor / nev_resolution)
        vn_avg[:, 0] = vn_avg[:, 0] / totalN
        vn_real = vn_real / nev_pT
        vn_imag = vn_imag / nev_pT
        vn_real_err = sqrt(vn_real_err / nev_pT - vn_real ** 2) / sqrt(
            nev_pT - 1)
        vn_imag_err = sqrt(vn_imag_err / nev_pT - vn_imag ** 2) / sqrt(
            nev_pT - 1)
        vn_avg[:, 1] = vn_real / resolutionFactor
        vn_avg[:, 2] = vn_real_err / resolutionFactor
        vn_avg[:, 3] = vn_imag / resolutionFactor
        vn_avg[:, 4] = vn_imag_err / resolutionFactor

        #interpolate results to desired pT range
        vn_real_interp = interp(pT_range, vn_avg[:, 0], vn_avg[:, 1])
        vn_real_interp_err = interp(pT_range, vn_avg[:, 0], vn_avg[:, 2])
        vn_imag_interp = interp(pT_range, vn_avg[:, 0], vn_avg[:, 3])
        vn_imag_interp_err = interp(pT_range, vn_avg[:, 0], vn_avg[:, 4])
        results = array([pT_range, vn_real_interp, vn_real_interp_err,
                         vn_imag_interp, vn_imag_interp_err])
        return transpose(results)

    def get_intevn_flow(
            self, particle_name, method, order, pT_range=(0.0, 3.0)):
        """
            compute pT integrated event plane or scalar product vn 
            averaged over all events. 
            The pT integrated range is given by the user
        """
        if method == 'event_plane':
            print("collect pT integraged vn{EP} of %s, pT range from "
                  "(%g, %g)..." % (particle_name, pT_range[0], pT_range[1]))
        elif method == 'scalar_product':
            print("collect pT integraged vn{SP} of %s, pT range from "
                  "(%g, %g)..." % (particle_name, pT_range[0], pT_range[1]))
        else:
            raise ValueError("AnalyzedDataReader.get_diffvn_flow: "
                             "invalid method : %s" % method)

        pid = self.pid_lookup[particle_name]
        analyzed_table_name_diff = 'flow_Qn_vectors_pTdiff'
        analyzed_table_name_inte = 'flow_Qn_vectors'

        vn_avg = zeros(5)
        vn_real = 0.0
        vn_imag = 0.0
        vn_real_err = 0.0
        vn_imag_err = 0.0
        totalN = 0
        nev = 0

        resolutionFactor = 0.0
        resolutionFactor_imag = 0.0
        nev_resolution = 0

        npT = len(self.db.executeSQLquery(
            "select pT from %s where hydro_event_id = %d and "
            "urqmd_event_id = %d and pid = %d and weight_type = '1' and "
            "n = %d and (%g <= pT and pT <= %g)"
            % (analyzed_table_name_diff, 1, 1, pid, order, pT_range[0],
               pT_range[1])
        ).fetchall())

        #fetch data
        for ibin in range(1, self.nev_bin):
            print("processing events %d to %d ..."
                  % ((ibin - 1) * self.process_nev, ibin * self.process_nev))
            hydro_ev_bound_low = self.event_bound_hydro[ibin - 1]
            hydro_ev_bound_high = self.event_bound_hydro[ibin]
            urqmd_ev_bound_low = self.event_bound_urqmd[ibin - 1]
            urqmd_ev_bound_high = self.event_bound_urqmd[ibin]
            if hydro_ev_bound_low == hydro_ev_bound_high:
                temp_data = array(self.db.executeSQLquery(
                    "select pT, Nparticle, Qn_real, Qn_imag from %s "
                    "where pid = %d and weight_type = '1' and n = %d and "
                    "hydro_event_id = %d and "
                    "(%d <= urqmd_event_id and urqmd_event_id < %d) and "
                    "(%g <= pT and pT <= %g)"
                    % (analyzed_table_name_diff, pid, order,
                       hydro_ev_bound_low, urqmd_ev_bound_low,
                       urqmd_ev_bound_high, pT_range[0], pT_range[1])
                ).fetchall())
                ref_data = array(self.db.executeSQLquery(
                    "select QnA_real, QnA_imag, QnB_real, QnB_imag, "
                    "QnC_real, QnC_imag, QnD_real, QnD_imag from %s "
                    "where pid = 1 and weight_type = '1' and n = %d and "
                    "hydro_event_id = %d and "
                    "(%d <= urqmd_event_id and urqmd_event_id < %d)"
                    % (analyzed_table_name_inte, order, hydro_ev_bound_low,
                       urqmd_ev_bound_low, urqmd_ev_bound_high)
                ).fetchall())
            else:
                temp_data = array(self.db.executeSQLquery(
                    "select pT, Nparticle, Qn_real, Qn_imag from %s "
                    "where pid = %d and weight_type = '1' and n = %d and "
                    "((hydro_event_id = %d and urqmd_event_id >= %d) "
                    " or (%d < hydro_event_id and hydro_event_id < %d) "
                    " or (hydro_event_id = %d and urqmd_event_id < %d)) and "
                    "(%g <= pT and pT <= %g)"
                    % (analyzed_table_name_diff, pid, order,
                       hydro_ev_bound_low, urqmd_ev_bound_low,
                       hydro_ev_bound_low, hydro_ev_bound_high,
                       hydro_ev_bound_high, urqmd_ev_bound_high,
                       pT_range[0], pT_range[1])
                ).fetchall())
                ref_data = array(self.db.executeSQLquery(
                    "select QnA_real, QnA_imag, QnB_real, QnB_imag, "
                    "QnC_real, QnC_imag, QnD_real, QnD_imag from %s "
                    "where pid = 1 and weight_type = '1' and n = %d and "
                    "((hydro_event_id = %d and urqmd_event_id >= %d) "
                    " or (%d < hydro_event_id and hydro_event_id < %d) "
                    " or (hydro_event_id = %d and urqmd_event_id < %d))"
                    % (analyzed_table_name_inte, order, hydro_ev_bound_low,
                       urqmd_ev_bound_low, hydro_ev_bound_low,
                       hydro_ev_bound_high, hydro_ev_bound_high,
                       urqmd_ev_bound_high)
                ).fetchall())
            ref_Qn_AC_real = 0.5 * (ref_data[:, 0] + ref_data[:, 4])
            ref_Qn_AC_imag = 0.5 * (ref_data[:, 1] + ref_data[:, 5])
            ref_Qn_BD_real = 0.5 * (ref_data[:, 2] + ref_data[:, 6])
            ref_Qn_BD_imag = 0.5 * (ref_data[:, 3] + ref_data[:, 7])
            ref_QnAC = sqrt(ref_Qn_AC_real ** 2. + ref_Qn_AC_imag ** 2)
            ref_QnBD = sqrt(ref_Qn_BD_real ** 2. + ref_Qn_BD_imag ** 2)

            if method == 'event_plane':
                resolutionFactor += sum((ref_Qn_AC_real * ref_Qn_BD_real
                                         + ref_Qn_AC_imag * ref_Qn_BD_imag)
                                        / ref_QnAC / ref_QnBD)
                resolutionFactor_imag += (
                    sum((ref_Qn_AC_imag * ref_Qn_BD_real
                         - ref_Qn_AC_real * ref_Qn_BD_imag)
                        / ref_QnAC / ref_QnBD))
            else:
                resolutionFactor += sum(ref_Qn_AC_real * ref_Qn_BD_real
                                        + ref_Qn_AC_imag * ref_Qn_BD_imag)
                resolutionFactor_imag += sum(ref_Qn_AC_imag * ref_Qn_BD_real
                                             - ref_Qn_AC_real * ref_Qn_BD_imag)
            nev_resolution += len(ref_data[:, 0])
            vn_avg[0] += sum(temp_data[:, 0] * temp_data[:, 1])  #<pT>
            totalN += sum(temp_data[:, 1])
            temp_nev = int(len(temp_data[:, 0]) / npT)
            for iev in range(temp_nev):
                ev_data = temp_data[iev * npT:(iev + 1) * npT, :]
                nparticle = sum(ev_data[:, 1])
                if nparticle == 0: continue
                nev += 1
                pTinte_Qn_x = sum(ev_data[:, 1] * ev_data[:, 2]) / nparticle
                pTinte_Qn_y = sum(ev_data[:, 1] * ev_data[:, 3]) / nparticle
                if method == 'event_plane':
                    temp_real = ((pTinte_Qn_x * ref_Qn_AC_real[iev]
                                  + pTinte_Qn_y * ref_Qn_AC_imag[iev])
                                 / ref_QnAC[iev])
                    temp_imag = ((pTinte_Qn_y * ref_Qn_AC_real[iev]
                                  - pTinte_Qn_x * ref_Qn_AC_imag[iev])
                                 / ref_QnAC[iev])
                else:
                    temp_real = (pTinte_Qn_x * ref_Qn_AC_real[iev]
                                 + pTinte_Qn_y * ref_Qn_AC_imag[iev])
                    temp_imag = (pTinte_Qn_y * ref_Qn_AC_real[iev]
                                 - pTinte_Qn_x * ref_Qn_AC_imag[iev])
                vn_real += temp_real
                vn_imag += temp_imag
                vn_real_err += temp_real ** 2.
                vn_imag_err += temp_imag ** 2.
        resolutionFactor = sqrt(resolutionFactor / nev_resolution)
        resolutionFactor_imag = resolutionFactor_imag / nev_resolution
        vn_avg[0] = vn_avg[0] / totalN
        vn_real = vn_real / nev
        vn_imag = vn_imag / nev
        vn_real_err = sqrt(vn_real_err / nev - vn_real ** 2) / sqrt(nev - 1)
        vn_imag_err = sqrt(vn_imag_err / nev - vn_imag ** 2) / sqrt(nev - 1)
        vn_avg[1] = vn_real / resolutionFactor
        vn_avg[2] = vn_real_err / resolutionFactor
        vn_avg[3] = vn_imag / resolutionFactor
        vn_avg[4] = vn_imag_err / resolutionFactor

        return (vn_avg)


    def get_particle_meanPT(
            self, particle_name):
        """
            compute mean pT of user specified particle. The rapidity 
            range has been constraint on -0.5<=rapidity<=0.5 when 
            collecting collecting pt spectra for identified particles.
        """
        print("read mean pT for %s ... " % particle_name)
        pid = self.pid_lookup[particle_name]

        analyzed_table_name = 'particle_meanPT'
        try:
            pt_data = self.db.executeSQLquery(
                "select mean_pT_value, mean_pT_error from %s "
                "where pid=%d"%(analyzed_table_name, pid)).fetchall()
        except:
            print "Cannnot read from table %s"%analyzed_table_name

        return array(pt_data)
        # # in accordance with ALICE rapidity cut (arXiv:1303.0737)
        # analyzed_table_name = 'particle_pT_spectra' # spectra on rapidity

        # meanpT_avg = array([0.0, 0.0])
        # totalN = 0.0
        # #fetch data
        # for ibin in range(1, self.nev_bin):
        #     print("processing events %d to %d ..."
        #           % ((ibin - 1) * self.process_nev, ibin * self.process_nev))
        #     hydro_ev_bound_low = self.event_bound_hydro[ibin - 1]
        #     hydro_ev_bound_high = self.event_bound_hydro[ibin]
        #     urqmd_ev_bound_low = self.event_bound_urqmd[ibin - 1]
        #     urqmd_ev_bound_high = self.event_bound_urqmd[ibin]
        #     if hydro_ev_bound_low == hydro_ev_bound_high:
        #         temp_data = array(self.db.executeSQLquery(
        #             "select pT, dN_dydpT from %s where pid = %d and "
        #             "hydro_event_id = %d and (%d <= urqmd_event_id "
        #             "and urqmd_event_id < %d)"
        #             % (analyzed_table_name, pid, hydro_ev_bound_low,
        #                urqmd_ev_bound_low, urqmd_ev_bound_high)).fetchall())
        #     else:
        #         temp_data = array(self.db.executeSQLquery(
        #             "select pT, dN_dydpT from %s where pid = %d and "
        #             "((hydro_event_id = %d and urqmd_event_id >= %d) "
        #             " or (%d < hydro_event_id and hydro_event_id < %d) "
        #             " or (hydro_event_id = %d and urqmd_event_id < %d))"
        #             % (analyzed_table_name, pid, hydro_ev_bound_low,
        #                urqmd_ev_bound_low, hydro_ev_bound_low,
        #                hydro_ev_bound_high, hydro_ev_bound_high,
        #                urqmd_ev_bound_high)).fetchall())
        #     # calculate total pT, total pT^2 and total particle number for one event
        #     meanpT_avg[0] += sum(temp_data[::, 0] * temp_data[::, 1])
        #     meanpT_avg[1] += sum(temp_data[::, 0]**2 * temp_data[::, 1])
        #     totalN += sum(temp_data[::, 1])

        # # calculate mean pT and <pT>_err
        # meanpT_avg[0] = meanpT_avg[0] / totalN
        # meanpT_avg[1] = (sqrt(meanpT_avg[1] / totalN - meanpT_avg[0] ** 2)
        #                 / sqrt(self.tot_nev - 1))
        # return meanpT_avg

    def get_diffvn_2pc_flow(
            self, particle_name, order, pT_range=linspace(0.0, 3.0, 31)):
        """
            compute pT differential vn[2](pT)
            the results are interpolated to desired pT points given by the user
            Note: since both particles are taken from the same bin, one needs
                  to subtract self-correlation when multiply Qn with its
                  complex conjugate
        """
        print("collect pT differential vn[2](pT) of %s ..." % particle_name)
        eps = 1e-15
        pid = self.pid_lookup[particle_name]
        analyzed_table_name_diff = 'flow_Qn_vectors_pTdiff'

        npT = len(self.db.executeSQLquery(
            "select pT from %s where hydro_event_id = %d and "
            "urqmd_event_id = %d and pid = %d and weight_type = '1' and n = %d"
            % (analyzed_table_name_diff, 1, 1, pid, order)).fetchall())

        vn_avg = zeros([npT, 3])
        vn_real = zeros(npT)
        vn_imag = zeros(npT)
        vn_real_err = zeros(npT)
        vn_imag_err = zeros(npT)
        totalN = zeros(npT)
        nev_pT = zeros(npT)

        #fetch data
        for ibin in range(1, self.nev_bin):
            print("processing events %d to %d ..."
                  % ((ibin - 1) * self.process_nev, ibin * self.process_nev))
            hydro_ev_bound_low = self.event_bound_hydro[ibin - 1]
            hydro_ev_bound_high = self.event_bound_hydro[ibin]
            urqmd_ev_bound_low = self.event_bound_urqmd[ibin - 1]
            urqmd_ev_bound_high = self.event_bound_urqmd[ibin]
            if hydro_ev_bound_low == hydro_ev_bound_high:
                temp_data = array(self.db.executeSQLquery(
                    "select pT, Nparticle, Qn_real, Qn_imag from %s "
                    "where pid = %d and weight_type = '1' and n = %d and "
                    "hydro_event_id = %d and "
                    "(%d <= urqmd_event_id and urqmd_event_id < %d)"
                    % (analyzed_table_name_diff, pid, order,
                       hydro_ev_bound_low, urqmd_ev_bound_low,
                       urqmd_ev_bound_high)
                ).fetchall())
            else:
                temp_data = array(self.db.executeSQLquery(
                    "select pT, Nparticle, Qn_real, Qn_imag from %s "
                    "where pid = %d and weight_type = '1' and n = %d and "
                    "((hydro_event_id = %d and urqmd_event_id >= %d) "
                    " or (%d < hydro_event_id and hydro_event_id < %d) "
                    " or (hydro_event_id = %d and urqmd_event_id < %d))"
                    % (analyzed_table_name_diff, pid, order,
                       hydro_ev_bound_low, urqmd_ev_bound_low,
                       hydro_ev_bound_low, hydro_ev_bound_high,
                       hydro_ev_bound_high, urqmd_ev_bound_high)
                ).fetchall())
            for ipT in range(npT):
                single_pT_data = temp_data[ipT::npT, :]
                vn_avg[ipT, 0] += sum(
                    single_pT_data[:, 0] * single_pT_data[:, 1])
                totalN[ipT] += sum(single_pT_data[:, 1])

                effective_ev = single_pT_data[single_pT_data[:, 1] > 1, 1:4]
                nev_pT[ipT] += len(effective_ev[:, 0])
                temp_array = ((effective_ev[:, 0]
                               * (
                    effective_ev[:, 1] ** 2 + effective_ev[:, 2] ** 2) - 1.)
                              / (effective_ev[:, 0] - 1))
                vn_real[ipT] += sum(temp_array)
                vn_real_err[ipT] += sum(temp_array ** 2.)

        vn_avg[:, 0] = vn_avg[:, 0] / totalN
        vn_real = vn_real / nev_pT
        vn_real_err = sqrt(vn_real_err / nev_pT - vn_real ** 2) / sqrt(
            nev_pT - 1)
        vn_avg[:, 1] = sqrt(vn_real)
        vn_avg[:, 2] = vn_real_err / 2. / sqrt(vn_real)

        #interpolate results to desired pT range
        vn_avg_interp = interp(pT_range, vn_avg[:, 0], vn_avg[:, 1])
        vn_avg_interp_err = interp(pT_range, vn_avg[:, 0], vn_avg[:, 2])
        results = array([pT_range, vn_avg_interp, vn_avg_interp_err])
        return transpose(results)

    def get_intevn_2pc_flow(
            self, particle_name, order, pT_range=(0.0, 3.0)):
        """
            compute pT integrated event plane vn[2] averaged over 
            all events the pT integrated range is given by the user
        """
        print("collect pT integraged vn[2] of %s, pT range from (%g, %g) GeV"
              % (particle_name, pT_range[0], pT_range[1]))
        pid = self.pid_lookup[particle_name]
        analyzed_table_name_diff = 'flow_Qn_vectors_pTdiff'

        vn_avg = zeros(3)
        vn_real = 0.0
        vn_imag = 0.0
        vn_real_err = 0.0
        vn_imag_err = 0.0
        totalN = 0
        nev = 0

        npT = len(self.db.executeSQLquery(
            "select pT from %s where hydro_event_id = %d and "
            "urqmd_event_id = %d and pid = %d and weight_type = '1' and "
            "n = %d and (%g <= pT and pT <= %g)"
            % (analyzed_table_name_diff, 1, 1, pid, order, pT_range[0],
               pT_range[1])
        ).fetchall())

        #fetch data
        for ibin in range(1, self.nev_bin):
            print("processing events %d to %d ..."
                  % ((ibin - 1) * self.process_nev, ibin * self.process_nev))
            hydro_ev_bound_low = self.event_bound_hydro[ibin - 1]
            hydro_ev_bound_high = self.event_bound_hydro[ibin]
            urqmd_ev_bound_low = self.event_bound_urqmd[ibin - 1]
            urqmd_ev_bound_high = self.event_bound_urqmd[ibin]
            if hydro_ev_bound_low == hydro_ev_bound_high:
                temp_data = array(self.db.executeSQLquery(
                    "select pT, Nparticle, Qn_real, Qn_imag from %s "
                    "where pid = %d and weight_type = '1' and n = %d and "
                    "hydro_event_id = %d and "
                    "(%d <= urqmd_event_id and urqmd_event_id < %d) and "
                    "(%g <= pT and pT <= %g)"
                    % (analyzed_table_name_diff, pid, order,
                       hydro_ev_bound_low, urqmd_ev_bound_low,
                       urqmd_ev_bound_high, pT_range[0], pT_range[1])
                ).fetchall())
            else:
                temp_data = array(self.db.executeSQLquery(
                    "select pT, Nparticle, Qn_real, Qn_imag from %s "
                    "where pid = %d and weight_type = '1' and n = %d and "
                    "((hydro_event_id = %d and urqmd_event_id >= %d) "
                    " or (%d < hydro_event_id and hydro_event_id < %d) "
                    " or (hydro_event_id = %d and urqmd_event_id < %d)) and "
                    "(%g <= pT and pT <= %g)"
                    % (analyzed_table_name_diff, pid, order,
                       hydro_ev_bound_low, urqmd_ev_bound_low,
                       hydro_ev_bound_low, hydro_ev_bound_high,
                       hydro_ev_bound_high, urqmd_ev_bound_high,
                       pT_range[0], pT_range[1])
                ).fetchall())
            vn_avg[0] += sum(temp_data[:, 0] * temp_data[:, 1])  #<pT>
            totalN += sum(temp_data[:, 1])
            temp_nev = int(len(temp_data[:, 0]) / npT)
            for iev in range(temp_nev):
                ev_data = temp_data[iev * npT:(iev + 1) * npT, :]
                nparticle = sum(ev_data[:, 1])
                if nparticle <= 1: continue
                nev += 1
                pTinte_Qn_x = sum(ev_data[:, 1] * ev_data[:, 2]) / nparticle
                pTinte_Qn_y = sum(ev_data[:, 1] * ev_data[:, 3]) / nparticle

                temp_real = (
                    (nparticle * (pTinte_Qn_x ** 2 + pTinte_Qn_y ** 2) - 1)
                    / (nparticle - 1))
                vn_real += temp_real
                vn_real_err += temp_real ** 2.
        vn_avg[0] = vn_avg[0] / totalN
        vn_real = vn_real / nev
        vn_real_err = sqrt(vn_real_err / nev - vn_real ** 2) / sqrt(nev - 1)
        vn_avg[1] = sqrt(vn_real)
        vn_avg[2] = vn_real_err / 2. / sqrt(vn_real)

        return (vn_avg)

    #    def getFullplaneResolutionFactor(self, resolutionFactor_sub, Nfactor):
    #        """
    #            use binary search for numerical solution for
    #            R(chi_s) = resolutionFactor_sub
    #            where R(chi) is defined in Eq. (7) in arXiv:0904.2315v3
    #            and calculate resolutionFactor for the full event
    #        """
    #        # check
    #        if(resolutionFactor_sub > 1.0):
    #            print("error: resolutionFactor_sub = % g,  is larger than 1!"
    #                  % resolutionFactor_sub)
    #            exit(-1)
    #
    #        tol = 1e-8 # accuracy
    #
    #        #search boundary
    #        left = 0.0; right = 2.0 # R(2.0) > 1.0
    #        mid = (right + left)*0.5
    #        dis = right - left
    #        while dis > tol:
    #            midval = self.resolution_Rfunction(mid)
    #            diff = resolutionFactor_sub - midval
    #            if abs(diff) < tol:
    #                chi_s = mid
    #                break
    #            elif diff > tol :
    #                left = mid
    #            else:
    #                right = mid
    #            dis = right - left
    #            mid = (right + left)*0.5
    #        chi_s = mid
    #        return(self.resolution_Rfunction(chi_s*sqrt(Nfactor)))
    #
    #    def resolution_Rfunction(self, chi):
    #        """
    #            R(chi) for calculating resolution factor for the full event
    #        """
    #        chisq = chi*chi
    #        result = (sqrt(pi)/2*exp(-chisq/2)*chi*(scipy.special.i0(chisq/2)
    #                  + scipy.special.i1(chisq/2)))
    #        return result

    def get_ptinte_two_flow_correlation(
            self, particle_name, method, n1, n2, c1=1, c2=1,
            pT_1_range=(0.0, 5.0), pT_2_range=(0.0, 5.0)):
        """
            get pT integrated two flow vectors correlations according to event
            plane method or scalar product method
            event plane method:
            r_{n1,n2} = <(Q_n1/|Q_n1|)^c1*conj((Q_n2/|Q_n2|)^c2)>_ev
                        /sqrt(<(Q_n1A/|Q_n1A|*conj(Q_n1B/|Q_n1B|))^c1>_ev*
                              *<(Q_n2A/|Q_n2A|*conj(Q_n2B/|Q_n2B|))^c2>_ev)
            scalar product:
            r_{n1,n2} = <(Q_n1)^c1*conj((Q_n2)^c2)>_ev
                        /sqrt(<(Q_n1*conj(Q_n1))^c1>_ev*
                              *<(Q_n2*conj(Q_n2))^c2>_ev)
            Q_n1 and Q_n2 are take from two subevents with an eta gap = 1
            at forward and backward rapidity (B+D) and (A+C)

            This function will return 
                (rn_real, rn_real_err, rn_imag, rn_imag_err)

            Note: if n1 = n2, r_{n1, n2} reduces to flow factorization ratio
        """
        #if n1*c1 + n2*c2 != 0:
        #    raise ValueError(
        #        "AnalyzedDataReader.get_ptinte_two_flow_correlation_ep: "
        #        "n1*c1 + n2*c2 = %d != 0!" % (n1*c1 + n2*c2))
        if method not in ['event_plane', 'scalar_product']:
            raise ValueError(
                "AnalyzedDataReader.get_ptinte_two_flow_correlation: "
                "invalid method : %s" % method)
        print("collect pT integrated flow two-plane correlation "
              "according to %s method ..." % method)
        print("r_{%d*%d, %d*%d} of %s, pT_1 range = (%g, %g) GeV and "
              "pT_2 range = (%g, %g) GeV"
              % (c1, n1, c2, n2, particle_name, pT_1_range[0], pT_1_range[1],
                 pT_2_range[0], pT_2_range[1]))
        n1 = abs(n1)
        n2 = abs(n2)
        pid = self.pid_lookup[particle_name]
        analyzed_table_name_diff = 'flow_Qn_vectors_pTdiff'

        rn_avg = zeros(4)
        rn_real = 0.0
        rn_imag = 0.0
        rn_real_err = 0.0
        rn_imag_err = 0.0
        totalN = 0
        nev = 0

        resolutionFactor_1 = 0.0
        resolutionFactor_2 = 0.0

        npT_1 = len(self.db.executeSQLquery(
            "select pT from %s where hydro_event_id = %d and "
            "urqmd_event_id = %d and pid = %d and weight_type = '1' and "
            "n = %d and (%g <= pT and pT <= %g)"
            % (analyzed_table_name_diff, 1, 1, pid, n1, pT_1_range[0],
               pT_1_range[1])
        ).fetchall())
        npT_2 = len(self.db.executeSQLquery(
            "select pT from %s where hydro_event_id = %d and "
            "urqmd_event_id = %d and pid = %d and weight_type = '1' and "
            "n = %d and (%g <= pT and pT <= %g)"
            % (analyzed_table_name_diff, 1, 1, pid, n2, pT_2_range[0],
               pT_2_range[1])
        ).fetchall())

        #fetch data
        for ibin in range(1, self.nev_bin):
            print("processing events %d to %d ..."
                  % ((ibin - 1) * self.process_nev, ibin * self.process_nev))
            hydro_ev_bound_low = self.event_bound_hydro[ibin - 1]
            hydro_ev_bound_high = self.event_bound_hydro[ibin]
            urqmd_ev_bound_low = self.event_bound_urqmd[ibin - 1]
            urqmd_ev_bound_high = self.event_bound_urqmd[ibin]
            if hydro_ev_bound_low == hydro_ev_bound_high:
                temp_1_data = array(self.db.executeSQLquery(
                    "select pT, Nparticle_sub, QnA_real, QnA_imag, "
                    "QnC_real, QnC_imag, QnB_real, QnB_imag, QnD_real, "
                    "QnD_imag from %s "
                    "where pid = %d and weight_type = '1' and n = %d and "
                    "hydro_event_id = %d and "
                    "(%d <= urqmd_event_id and urqmd_event_id < %d) and "
                    "(%g <= pT and pT <= %g)"
                    % (analyzed_table_name_diff, pid, n1,
                       hydro_ev_bound_low, urqmd_ev_bound_low,
                       urqmd_ev_bound_high, pT_1_range[0], pT_1_range[1])
                ).fetchall())
                temp_2_data = array(self.db.executeSQLquery(
                    "select pT, Nparticle_sub, QnB_real, QnB_imag, "
                    "QnD_real, QnD_imag, QnA_real, QnA_imag, QnC_real, "
                    "QnC_imag from %s "
                    "where pid = %d and weight_type = '1' and n = %d and "
                    "hydro_event_id = %d and "
                    "(%d <= urqmd_event_id and urqmd_event_id < %d) and "
                    "(%g <= pT and pT <= %g)"
                    % (analyzed_table_name_diff, pid, n2,
                       hydro_ev_bound_low, urqmd_ev_bound_low,
                       urqmd_ev_bound_high, pT_2_range[0], pT_2_range[1])
                ).fetchall())
            else:
                temp_1_data = array(self.db.executeSQLquery(
                    "select pT, Nparticle_sub, QnA_real, QnA_imag, "
                    "QnC_real, QnC_imag, QnB_real, QnB_imag, QnD_real, "
                    "QnD_imag from %s "
                    "where pid = %d and weight_type = '1' and n = %d and "
                    "((hydro_event_id = %d and urqmd_event_id >= %d) "
                    " or (%d < hydro_event_id and hydro_event_id < %d) "
                    " or (hydro_event_id = %d and urqmd_event_id < %d)) and "
                    "(%g <= pT and pT <= %g)"
                    % (analyzed_table_name_diff, pid, n1,
                       hydro_ev_bound_low, urqmd_ev_bound_low,
                       hydro_ev_bound_low, hydro_ev_bound_high,
                       hydro_ev_bound_high, urqmd_ev_bound_high,
                       pT_1_range[0], pT_1_range[1])
                ).fetchall())
                temp_2_data = array(self.db.executeSQLquery(
                    "select pT, Nparticle_sub, QnB_real, QnB_imag, "
                    "QnD_real, QnD_imag, QnA_real, QnA_imag, QnC_real, "
                    "QnC_imag from %s "
                    "where pid = %d and weight_type = '1' and n = %d and "
                    "((hydro_event_id = %d and urqmd_event_id >= %d) "
                    " or (%d < hydro_event_id and hydro_event_id < %d) "
                    " or (hydro_event_id = %d and urqmd_event_id < %d)) and "
                    "(%g <= pT and pT <= %g)"
                    % (analyzed_table_name_diff, pid, n2,
                       hydro_ev_bound_low, urqmd_ev_bound_low,
                       hydro_ev_bound_low, hydro_ev_bound_high,
                       hydro_ev_bound_high, urqmd_ev_bound_high,
                       pT_2_range[0], pT_2_range[1])
                ).fetchall())
            temp_nev = int(len(temp_1_data[:, 0]) / npT_1)
            for iev in range(temp_nev):
                ev_1_data = temp_1_data[iev * npT_1:(iev + 1) * npT_1, :]
                ev_2_data = temp_2_data[iev * npT_2:(iev + 1) * npT_2, :]
                nparticle_1 = sum(ev_1_data[:, 1])
                nparticle_2 = sum(ev_2_data[:, 1])
                if nparticle_1 < 1: continue
                if nparticle_2 < 1: continue
                nev += 1
                pTinte_Qn_1_x = (sum(ev_1_data[:, 1]
                                     * (ev_1_data[:, 2] + ev_1_data[:, 4])) / (
                                     2. * nparticle_1))
                pTinte_Qn_1_y = (sum(ev_1_data[:, 1]
                                     * (ev_1_data[:, 3] + ev_1_data[:, 5])) / (
                                     2. * nparticle_1))

                pTinte_Qn_ref_1_x = (sum(ev_1_data[:, 1]
                                         * (
                    ev_1_data[:, 6] + ev_1_data[:, 8])) / (2. * nparticle_1))
                pTinte_Qn_ref_1_y = (sum(ev_1_data[:, 1]
                                         * (
                    ev_1_data[:, 7] + ev_1_data[:, 9])) / (2. * nparticle_1))

                pTinte_Qn_1 = sqrt(pTinte_Qn_1_x ** 2. + pTinte_Qn_1_y ** 2)
                pTinte_Qn_1_psi = arctan2(pTinte_Qn_1_y, pTinte_Qn_1_x) / n1
                pTinte_Qn_ref_1 = (sqrt(pTinte_Qn_ref_1_x ** 2.
                                        + pTinte_Qn_ref_1_y ** 2))
                pTinte_Qn_ref_1_psi = (arctan2(pTinte_Qn_ref_1_y,
                                               pTinte_Qn_ref_1_x) / n1)

                pTinte_Qn_2_x = (sum(ev_2_data[:, 1]
                                     * (ev_2_data[:, 2] + ev_2_data[:, 4])) / (
                                     2. * nparticle_2))
                pTinte_Qn_2_y = (sum(ev_2_data[:, 1]
                                     * (ev_2_data[:, 3] + ev_2_data[:, 5])) / (
                                     2. * nparticle_2))

                pTinte_Qn_ref_2_x = (sum(ev_2_data[:, 1]
                                         * (
                    ev_2_data[:, 6] + ev_2_data[:, 8])) / (2. * nparticle_2))
                pTinte_Qn_ref_2_y = (sum(ev_2_data[:, 1]
                                         * (
                    ev_2_data[:, 7] + ev_2_data[:, 9])) / (2. * nparticle_2))

                pTinte_Qn_2 = sqrt(pTinte_Qn_2_x ** 2. + pTinte_Qn_2_y ** 2)
                pTinte_Qn_2_psi = arctan2(pTinte_Qn_2_y, pTinte_Qn_2_x) / n2

                pTinte_Qn_ref_2 = (sqrt(pTinte_Qn_ref_2_x ** 2.
                                        + pTinte_Qn_ref_2_y ** 2))
                pTinte_Qn_ref_2_psi = (arctan2(pTinte_Qn_ref_2_y,
                                               pTinte_Qn_ref_2_x) / n2)

                if method == 'event_plane':
                    temp_real = (cos(n1 * c1 * pTinte_Qn_1_psi
                                     - n2 * c2 * pTinte_Qn_2_psi)
                                 + cos(n1 * c1 * pTinte_Qn_ref_1_psi
                                       - n2 * c2 * pTinte_Qn_ref_2_psi))
                    temp_imag = (sin(n1 * c1 * pTinte_Qn_1_psi
                                     - n2 * c2 * pTinte_Qn_2_psi)
                                 + sin(n1 * c1 * pTinte_Qn_ref_1_psi
                                       - n2 * c2 * pTinte_Qn_ref_2_psi))
                    resolutionFactor_1 += cos(n1 * c1 * (pTinte_Qn_ref_1_psi
                                                         - pTinte_Qn_1_psi))
                    resolutionFactor_2 += cos(n2 * c2 * (pTinte_Qn_2_psi
                                                         - pTinte_Qn_ref_2_psi))
                else:
                    temp_real = (pTinte_Qn_1 ** c1 * pTinte_Qn_2 ** c2
                                 * cos(n1 * c1 * pTinte_Qn_1_psi
                                       - n2 * c2 * pTinte_Qn_2_psi)
                                 + pTinte_Qn_ref_1 ** c1 * pTinte_Qn_ref_2 ** c2
                                 * cos(n1 * c1 * pTinte_Qn_ref_1_psi
                                       - n2 * c2 * pTinte_Qn_ref_2_psi)
                    )
                    temp_imag = (pTinte_Qn_1 ** c1 * pTinte_Qn_2 ** c2
                                 * sin(n1 * c1 * pTinte_Qn_1_psi
                                       - n2 * c2 * pTinte_Qn_2_psi)
                                 + pTinte_Qn_ref_1 ** c1 * pTinte_Qn_ref_2 ** c2
                                 * sin(n1 * c1 * pTinte_Qn_ref_1_psi
                                       - n2 * c2 * pTinte_Qn_ref_2_psi)
                    )
                    resolutionFactor_1 += (
                        pTinte_Qn_1 ** c1 * pTinte_Qn_ref_1 ** c1
                        * cos(n1 * c1 * (pTinte_Qn_1_psi
                                         - pTinte_Qn_ref_1_psi)))
                    resolutionFactor_2 += (
                        pTinte_Qn_2 ** c2 * pTinte_Qn_ref_2 ** c2
                        * cos(n2 * c2 * (pTinte_Qn_2_psi
                                         - pTinte_Qn_ref_2_psi)))
                rn_real += temp_real
                rn_real_err += temp_real ** 2.
                rn_imag += temp_imag
                rn_imag_err += temp_imag ** 2.

        resolutionFactor_1 = sqrt(resolutionFactor_1 / nev)
        resolutionFactor_2 = sqrt(resolutionFactor_2 / nev)
        rn_real = rn_real / nev
        rn_imag = rn_imag / nev
        rn_real_err = sqrt(rn_real_err / nev - rn_real ** 2) / sqrt(nev - 1)
        rn_imag_err = sqrt(rn_imag_err / nev - rn_imag ** 2) / sqrt(nev - 1)

        rn_avg[0] = rn_real / (2. * resolutionFactor_1 * resolutionFactor_2)
        rn_avg[1] = rn_real_err / (
            2. * resolutionFactor_1 * resolutionFactor_2)
        rn_avg[2] = rn_imag / (2. * resolutionFactor_1 * resolutionFactor_2)
        rn_avg[3] = rn_imag_err / (
            2. * resolutionFactor_1 * resolutionFactor_2)

        return rn_avg

    def get_ptinte_three_flow_correlation(
            self, particle_name, method, n1, n2, n3, c1=1, c2=1, c3=1,
            pT_1_range=(0.0, 5.0), pT_2_range=(0.0, 5.0),
            pT_3_range=(0.0, 5.0)):
        """
            get pT integrated three flow vectors correlations according 
            to event plane or scalar product method
            r_{n1, n2, n3} = 
                <(Q_n1/|Q_n1|)^c1*(Q_n2/|Q_n2|)^c2*(Q_n3/|Q_n3|)^c3>_ev
                /(sqrt(<(Q_nA1/|Q_nA1|*conj(Q_nB1/|Q_nB1|))^c1>_ev)
                  *sqrt(<(Q_nA2/|Q_nA2|*conj(Q_nB2/|Q_nB2|))^c2>_ev)
                  *sqrt(<(Q_nA3/|Q_nA3|*conj(Q_nB3/|Q_nB3|))^c3>_ev))
            Q_n1 and Q_n2 are taken from two subevents at forward and backward
            rapidities (-1.5 < rap < -0.5 and 0.5 < rap < 1.5)
            Q_n3 is taken at mid-rapidity -2.5 <= rap < -1.5

            This function will return 
                (rn_real, rn_real_err, rn_imag, rn_imag_err)
        """
        if n1 * c1 + n2 * c2 + n3 * c3 != 0:
            raise ValueError(
                "AnalyzedDataReader.get_ptinte_three_flow_correlation: "
                "n1*c1 + n2*c2 + n3*c3 = %d != 0!" % (
                    n1 * c1 + n2 * c2 + n3 * c3))
        if method not in ['event_plane', 'scalar_product']:
            raise ValueError(
                "AnalyzedDataReader.get_ptinte_three_flow_correlation: "
                "invalid method : %s" % method)
        print("collect pT integrated flow three-plane correlation "
              "according to %s method ..." % method)
        print("r_{%d*%d, %d*%d, %d*%d} for %s \n"
              "with pT_1 range = (%g, %g) GeV and "
              "pT_2 range = (%g, %g) GeV and pT_3_range = (%g, %g) GeV"
              % (c1, n1, c2, n2, c3, n3, particle_name,
                 pT_1_range[0], pT_1_range[1], pT_2_range[0], pT_2_range[1],
                 pT_3_range[0], pT_3_range[1]))
        conjFlags = ones(3)
        if n1 < 0: conjFlags[0] = -1
        if n2 < 0: conjFlags[1] = -1
        if n3 < 0: conjFlags[2] = -1
        n1 = abs(n1)
        n2 = abs(n2)
        n3 = abs(n3)

        pid = self.pid_lookup[particle_name]
        analyzed_table_name_diff = 'flow_Qn_vectors_pTdiff'

        rn_avg = zeros(4)  # record final results
        rn_real = 0.0
        rn_imag = 0.0
        rn_real_err = 0.0
        rn_imag_err = 0.0
        totalN = 0
        nev = 0

        resolutionFactor_1_AB = 0.0
        resolutionFactor_1_CD = 0.0
        resolutionFactor_2_AB = 0.0
        resolutionFactor_2_CD = 0.0
        resolutionFactor_3_AB = 0.0
        resolutionFactor_3_CD = 0.0

        npT_1 = len(self.db.executeSQLquery(
            "select pT from %s where hydro_event_id = %d and "
            "urqmd_event_id = %d and pid = %d and weight_type = '1' and "
            "n = %d and (%g <= pT and pT <= %g)"
            % (analyzed_table_name_diff, 1, 1, pid, n1, pT_1_range[0],
               pT_1_range[1])
        ).fetchall())
        npT_2 = len(self.db.executeSQLquery(
            "select pT from %s where hydro_event_id = %d and "
            "urqmd_event_id = %d and pid = %d and weight_type = '1' and "
            "n = %d and (%g <= pT and pT <= %g)"
            % (analyzed_table_name_diff, 1, 1, pid, n2, pT_2_range[0],
               pT_2_range[1])
        ).fetchall())
        npT_3 = len(self.db.executeSQLquery(
            "select pT from %s where hydro_event_id = %d and "
            "urqmd_event_id = %d and pid = %d and weight_type = '1' and "
            "n = %d and (%g <= pT and pT <= %g)"
            % (analyzed_table_name_diff, 1, 1, pid, n3, pT_3_range[0],
               pT_3_range[1])
        ).fetchall())

        #fetch data
        for ibin in range(1, self.nev_bin):
            print("processing events %d to %d ..."
                  % ((ibin - 1) * self.process_nev, ibin * self.process_nev))
            hydro_ev_bound_low = self.event_bound_hydro[ibin - 1]
            hydro_ev_bound_high = self.event_bound_hydro[ibin]
            urqmd_ev_bound_low = self.event_bound_urqmd[ibin - 1]
            urqmd_ev_bound_high = self.event_bound_urqmd[ibin]
            if hydro_ev_bound_low == hydro_ev_bound_high:
                temp_1_data = array(self.db.executeSQLquery(
                    "select pT, Nparticle_sub, QnA_real, QnA_imag, "
                    "QnB_real, QnB_imag, QnC_real, QnC_imag, "
                    "QnD_real, QnD_imag from %s "
                    "where pid = %d and weight_type = '1' and n = %d and "
                    "hydro_event_id = %d and "
                    "(%d <= urqmd_event_id and urqmd_event_id < %d) and "
                    "(%g <= pT and pT <= %g)"
                    % (analyzed_table_name_diff, pid, n1,
                       hydro_ev_bound_low, urqmd_ev_bound_low,
                       urqmd_ev_bound_high, pT_1_range[0], pT_1_range[1])
                ).fetchall())
                temp_2_data = array(self.db.executeSQLquery(
                    "select pT, Nparticle_sub, QnA_real, QnA_imag, "
                    "QnB_real, QnB_imag, QnC_real, QnC_imag, "
                    "QnD_real, QnD_imag from %s "
                    "where pid = %d and weight_type = '1' and n = %d and "
                    "hydro_event_id = %d and "
                    "(%d <= urqmd_event_id and urqmd_event_id < %d) and "
                    "(%g <= pT and pT <= %g)"
                    % (analyzed_table_name_diff, pid, n2,
                       hydro_ev_bound_low, urqmd_ev_bound_low,
                       urqmd_ev_bound_high, pT_2_range[0], pT_2_range[1])
                ).fetchall())
                temp_3_data = array(self.db.executeSQLquery(
                    "select pT, Nparticle_sub, QnA_real, QnA_imag, "
                    "QnB_real, QnB_imag, QnC_real, QnC_imag, "
                    "QnD_real, QnD_imag from %s "
                    "where pid = %d and weight_type = '1' and n = %d and "
                    "hydro_event_id = %d and "
                    "(%d <= urqmd_event_id and urqmd_event_id < %d) and "
                    "(%g <= pT and pT <= %g)"
                    % (analyzed_table_name_diff, pid, n3,
                       hydro_ev_bound_low, urqmd_ev_bound_low,
                       urqmd_ev_bound_high, pT_3_range[0], pT_3_range[1])
                ).fetchall())
            else:
                temp_1_data = array(self.db.executeSQLquery(
                    "select pT, Nparticle_sub, QnA_real, QnA_imag, "
                    "QnB_real, QnB_imag, QnC_real, QnC_imag, "
                    "QnD_real, QnD_imag from %s "
                    "where pid = %d and weight_type = '1' and n = %d and "
                    "((hydro_event_id = %d and urqmd_event_id >= %d) "
                    " or (%d < hydro_event_id and hydro_event_id < %d) "
                    " or (hydro_event_id = %d and urqmd_event_id < %d)) and "
                    "(%g <= pT and pT <= %g)"
                    % (analyzed_table_name_diff, pid, n1,
                       hydro_ev_bound_low, urqmd_ev_bound_low,
                       hydro_ev_bound_low, hydro_ev_bound_high,
                       hydro_ev_bound_high, urqmd_ev_bound_high,
                       pT_1_range[0], pT_1_range[1])
                ).fetchall())
                temp_2_data = array(self.db.executeSQLquery(
                    "select pT, Nparticle_sub, QnA_real, QnA_imag, "
                    "QnB_real, QnB_imag, QnC_real, QnC_imag, "
                    "QnD_real, QnD_imag from %s "
                    "where pid = %d and weight_type = '1' and n = %d and "
                    "((hydro_event_id = %d and urqmd_event_id >= %d) "
                    " or (%d < hydro_event_id and hydro_event_id < %d) "
                    " or (hydro_event_id = %d and urqmd_event_id < %d)) and "
                    "(%g <= pT and pT <= %g)"
                    % (analyzed_table_name_diff, pid, n2,
                       hydro_ev_bound_low, urqmd_ev_bound_low,
                       hydro_ev_bound_low, hydro_ev_bound_high,
                       hydro_ev_bound_high, urqmd_ev_bound_high,
                       pT_2_range[0], pT_2_range[1])
                ).fetchall())
                temp_3_data = array(self.db.executeSQLquery(
                    "select pT, Nparticle_sub, QnA_real, QnA_imag, "
                    "QnB_real, QnB_imag, QnC_real, QnC_imag, "
                    "QnD_real, QnD_imag from %s "
                    "where pid = %d and weight_type = '1' and n = %d and "
                    "((hydro_event_id = %d and urqmd_event_id >= %d) "
                    " or (%d < hydro_event_id and hydro_event_id < %d) "
                    " or (hydro_event_id = %d and urqmd_event_id < %d)) and "
                    "(%g <= pT and pT <= %g)"
                    % (analyzed_table_name_diff, pid, n3,
                       hydro_ev_bound_low, urqmd_ev_bound_low,
                       hydro_ev_bound_low, hydro_ev_bound_high,
                       hydro_ev_bound_high, urqmd_ev_bound_high,
                       pT_3_range[0], pT_3_range[1])
                ).fetchall())
            temp_nev = int(len(temp_1_data[:, 0]) / npT_1)
            for iev in range(temp_nev):
                ev_1_data = temp_1_data[iev * npT_1:(iev + 1) * npT_1, :]
                ev_2_data = temp_2_data[iev * npT_2:(iev + 1) * npT_2, :]
                ev_3_data = temp_3_data[iev * npT_3:(iev + 1) * npT_3, :]
                nparticle_1 = sum(ev_1_data[:, 1])
                nparticle_2 = sum(ev_2_data[:, 1])
                nparticle_3 = sum(ev_3_data[:, 1])
                if nparticle_1 < 1: continue
                if nparticle_2 < 1: continue
                if nparticle_3 < 1: continue
                nev += 1

                pTinte_Qn1_A_x = sum(
                    ev_1_data[:, 1] * ev_1_data[:, 2]) / nparticle_1
                pTinte_Qn1_A_y = sum(
                    ev_1_data[:, 1] * ev_1_data[:, 3]) / nparticle_1
                pTinte_Qn1_B_x = sum(
                    ev_1_data[:, 1] * ev_1_data[:, 4]) / nparticle_1
                pTinte_Qn1_B_y = sum(
                    ev_1_data[:, 1] * ev_1_data[:, 5]) / nparticle_1
                pTinte_Qn1_C_x = sum(
                    ev_1_data[:, 1] * ev_1_data[:, 6]) / nparticle_1
                pTinte_Qn1_C_y = sum(
                    ev_1_data[:, 1] * ev_1_data[:, 7]) / nparticle_1
                pTinte_Qn1_D_x = sum(
                    ev_1_data[:, 1] * ev_1_data[:, 8]) / nparticle_1
                pTinte_Qn1_D_y = sum(
                    ev_1_data[:, 1] * ev_1_data[:, 9]) / nparticle_1

                pTinte_Qn1_A = sqrt(pTinte_Qn1_A_x ** 2. + pTinte_Qn1_A_y ** 2)
                pTinte_Qn1_B = sqrt(pTinte_Qn1_B_x ** 2. + pTinte_Qn1_B_y ** 2)
                pTinte_Qn1_C = sqrt(pTinte_Qn1_C_x ** 2. + pTinte_Qn1_C_y ** 2)
                pTinte_Qn1_D = sqrt(pTinte_Qn1_D_x ** 2. + pTinte_Qn1_D_y ** 2)
                pTinte_Qn1_A_psi = arctan2(pTinte_Qn1_A_y, pTinte_Qn1_A_x) / n1
                pTinte_Qn1_B_psi = arctan2(pTinte_Qn1_B_y, pTinte_Qn1_B_x) / n1
                pTinte_Qn1_C_psi = arctan2(pTinte_Qn1_C_y, pTinte_Qn1_C_x) / n1
                pTinte_Qn1_D_psi = arctan2(pTinte_Qn1_D_y, pTinte_Qn1_D_x) / n1

                pTinte_Qn2_A_x = sum(
                    ev_2_data[:, 1] * ev_2_data[:, 2]) / nparticle_2
                pTinte_Qn2_A_y = sum(
                    ev_2_data[:, 1] * ev_2_data[:, 3]) / nparticle_2
                pTinte_Qn2_B_x = sum(
                    ev_2_data[:, 1] * ev_2_data[:, 4]) / nparticle_2
                pTinte_Qn2_B_y = sum(
                    ev_2_data[:, 1] * ev_2_data[:, 5]) / nparticle_2
                pTinte_Qn2_C_x = sum(
                    ev_2_data[:, 1] * ev_2_data[:, 6]) / nparticle_2
                pTinte_Qn2_C_y = sum(
                    ev_2_data[:, 1] * ev_2_data[:, 7]) / nparticle_2
                pTinte_Qn2_D_x = sum(
                    ev_2_data[:, 1] * ev_2_data[:, 8]) / nparticle_2
                pTinte_Qn2_D_y = sum(
                    ev_2_data[:, 1] * ev_2_data[:, 9]) / nparticle_2

                pTinte_Qn2_A = sqrt(pTinte_Qn2_A_x ** 2. + pTinte_Qn2_A_y ** 2)
                pTinte_Qn2_B = sqrt(pTinte_Qn2_B_x ** 2. + pTinte_Qn2_B_y ** 2)
                pTinte_Qn2_C = sqrt(pTinte_Qn2_C_x ** 2. + pTinte_Qn2_C_y ** 2)
                pTinte_Qn2_D = sqrt(pTinte_Qn2_D_x ** 2. + pTinte_Qn2_D_y ** 2)
                pTinte_Qn2_A_psi = arctan2(pTinte_Qn2_A_y, pTinte_Qn2_A_x) / n2
                pTinte_Qn2_B_psi = arctan2(pTinte_Qn2_B_y, pTinte_Qn2_B_x) / n2
                pTinte_Qn2_C_psi = arctan2(pTinte_Qn2_C_y, pTinte_Qn2_C_x) / n2
                pTinte_Qn2_D_psi = arctan2(pTinte_Qn2_D_y, pTinte_Qn2_D_x) / n2

                pTinte_Qn3_A_x = sum(
                    ev_3_data[:, 1] * ev_3_data[:, 2]) / nparticle_3
                pTinte_Qn3_A_y = sum(
                    ev_3_data[:, 1] * ev_3_data[:, 3]) / nparticle_3
                pTinte_Qn3_B_x = sum(
                    ev_3_data[:, 1] * ev_3_data[:, 4]) / nparticle_3
                pTinte_Qn3_B_y = sum(
                    ev_3_data[:, 1] * ev_3_data[:, 5]) / nparticle_3
                pTinte_Qn3_C_x = sum(
                    ev_3_data[:, 1] * ev_3_data[:, 6]) / nparticle_3
                pTinte_Qn3_C_y = sum(
                    ev_3_data[:, 1] * ev_3_data[:, 7]) / nparticle_3
                pTinte_Qn3_D_x = sum(
                    ev_3_data[:, 1] * ev_3_data[:, 8]) / nparticle_3
                pTinte_Qn3_D_y = sum(
                    ev_3_data[:, 1] * ev_3_data[:, 9]) / nparticle_3

                pTinte_Qn3_A = sqrt(pTinte_Qn3_A_x ** 2. + pTinte_Qn3_A_y ** 2)
                pTinte_Qn3_B = sqrt(pTinte_Qn3_B_x ** 2. + pTinte_Qn3_B_y ** 2)
                pTinte_Qn3_C = sqrt(pTinte_Qn3_C_x ** 2. + pTinte_Qn3_C_y ** 2)
                pTinte_Qn3_D = sqrt(pTinte_Qn3_D_x ** 2. + pTinte_Qn3_D_y ** 2)
                pTinte_Qn3_A_psi = arctan2(pTinte_Qn3_A_y, pTinte_Qn3_A_x) / n3
                pTinte_Qn3_B_psi = arctan2(pTinte_Qn3_B_y, pTinte_Qn3_B_x) / n3
                pTinte_Qn3_C_psi = arctan2(pTinte_Qn3_C_y, pTinte_Qn3_C_x) / n3
                pTinte_Qn3_D_psi = arctan2(pTinte_Qn3_D_y, pTinte_Qn3_D_x) / n3

                if method == 'event_plane':
                    temp_real = (cos(conjFlags[0] * n1 * c1 * pTinte_Qn1_A_psi
                                     + conjFlags[
                        1] * n2 * c2 * pTinte_Qn2_B_psi
                                     + conjFlags[
                        2] * n3 * c3 * pTinte_Qn3_C_psi)
                                 + cos(
                        conjFlags[0] * n1 * c1 * pTinte_Qn1_B_psi
                        + conjFlags[1] * n2 * c2 * pTinte_Qn2_A_psi
                        + conjFlags[2] * n3 * c3 * pTinte_Qn3_C_psi)
                                 + cos(
                        conjFlags[0] * n1 * c1 * pTinte_Qn1_C_psi
                        + conjFlags[1] * n2 * c2 * pTinte_Qn2_A_psi
                        + conjFlags[2] * n3 * c3 * pTinte_Qn3_B_psi)
                                 + cos(
                        conjFlags[0] * n1 * c1 * pTinte_Qn1_C_psi
                        + conjFlags[1] * n2 * c2 * pTinte_Qn2_B_psi
                        + conjFlags[2] * n3 * c3 * pTinte_Qn3_A_psi)
                                 + cos(
                        conjFlags[0] * n1 * c1 * pTinte_Qn1_A_psi
                        + conjFlags[1] * n2 * c2 * pTinte_Qn2_C_psi
                        + conjFlags[2] * n3 * c3 * pTinte_Qn3_B_psi)
                                 + cos(
                        conjFlags[0] * n1 * c1 * pTinte_Qn1_B_psi
                        + conjFlags[1] * n2 * c2 * pTinte_Qn2_C_psi
                        + conjFlags[2] * n3 * c3 * pTinte_Qn3_A_psi)
                    )
                    temp_imag = (sin(conjFlags[0] * n1 * c1 * pTinte_Qn1_A_psi
                                     + conjFlags[
                        1] * n2 * c2 * pTinte_Qn2_B_psi
                                     + conjFlags[
                        2] * n3 * c3 * pTinte_Qn3_C_psi)
                                 + sin(
                        conjFlags[0] * n1 * c1 * pTinte_Qn1_B_psi
                        + conjFlags[1] * n2 * c2 * pTinte_Qn2_A_psi
                        + conjFlags[2] * n3 * c3 * pTinte_Qn3_C_psi)
                                 + sin(
                        conjFlags[0] * n1 * c1 * pTinte_Qn1_C_psi
                        + conjFlags[1] * n2 * c2 * pTinte_Qn2_A_psi
                        + conjFlags[2] * n3 * c3 * pTinte_Qn3_B_psi)
                                 + sin(
                        conjFlags[0] * n1 * c1 * pTinte_Qn1_C_psi
                        + conjFlags[1] * n2 * c2 * pTinte_Qn2_B_psi
                        + conjFlags[2] * n3 * c3 * pTinte_Qn3_A_psi)
                                 + sin(
                        conjFlags[0] * n1 * c1 * pTinte_Qn1_A_psi
                        + conjFlags[1] * n2 * c2 * pTinte_Qn2_C_psi
                        + conjFlags[2] * n3 * c3 * pTinte_Qn3_B_psi)
                                 + sin(
                        conjFlags[0] * n1 * c1 * pTinte_Qn1_B_psi
                        + conjFlags[1] * n2 * c2 * pTinte_Qn2_C_psi
                        + conjFlags[2] * n3 * c3 * pTinte_Qn3_A_psi)
                    )
                    resolutionFactor_1_AB += cos(n1 * c1 * (pTinte_Qn1_A_psi
                                                            - pTinte_Qn1_B_psi))
                    resolutionFactor_1_CD += cos(n1 * c1 * (pTinte_Qn1_C_psi
                                                            - pTinte_Qn1_D_psi))
                    resolutionFactor_2_AB += cos(n2 * c2 * (pTinte_Qn2_A_psi
                                                            - pTinte_Qn2_B_psi))
                    resolutionFactor_2_CD += cos(n2 * c2 * (pTinte_Qn2_C_psi
                                                            - pTinte_Qn2_D_psi))
                    resolutionFactor_3_AB += cos(n3 * c3 * (pTinte_Qn3_A_psi
                                                            - pTinte_Qn3_B_psi))
                    resolutionFactor_3_CD += cos(n3 * c3 * (pTinte_Qn3_C_psi
                                                            - pTinte_Qn3_D_psi))
                else:
                    temp_real = (
                        pTinte_Qn1_A ** c1 * pTinte_Qn2_B ** c2 * pTinte_Qn3_C ** c3
                        * cos(conjFlags[0] * n1 * c1 * pTinte_Qn1_A_psi
                              + conjFlags[1] * n2 * c2 * pTinte_Qn2_B_psi
                              + conjFlags[2] * n3 * c3 * pTinte_Qn3_C_psi)
                        + pTinte_Qn1_B ** c1 * pTinte_Qn2_A ** c2 * pTinte_Qn3_C ** c3
                        * cos(conjFlags[0] * n1 * c1 * pTinte_Qn1_B_psi
                              + conjFlags[1] * n2 * c2 * pTinte_Qn2_A_psi
                              + conjFlags[2] * n3 * c3 * pTinte_Qn3_C_psi)
                        + pTinte_Qn1_C ** c1 * pTinte_Qn2_A ** c2 * pTinte_Qn3_B ** c3
                        * cos(conjFlags[0] * n1 * c1 * pTinte_Qn1_C_psi
                              + conjFlags[1] * n2 * c2 * pTinte_Qn2_A_psi
                              + conjFlags[2] * n3 * c3 * pTinte_Qn3_B_psi)
                        + pTinte_Qn1_C ** c1 * pTinte_Qn2_B ** c2 * pTinte_Qn3_A ** c3
                        * cos(conjFlags[0] * n1 * c1 * pTinte_Qn1_C_psi
                              + conjFlags[1] * n2 * c2 * pTinte_Qn2_B_psi
                              + conjFlags[2] * n3 * c3 * pTinte_Qn3_A_psi)
                        + pTinte_Qn1_A ** c1 * pTinte_Qn2_C ** c2 * pTinte_Qn3_B ** c3
                        * cos(conjFlags[0] * n1 * c1 * pTinte_Qn1_A_psi
                              + conjFlags[1] * n2 * c2 * pTinte_Qn2_C_psi
                              + conjFlags[2] * n3 * c3 * pTinte_Qn3_B_psi)
                        + pTinte_Qn1_B ** c1 * pTinte_Qn2_C ** c2 * pTinte_Qn3_A ** c3
                        * cos(conjFlags[0] * n1 * c1 * pTinte_Qn1_B_psi
                              + conjFlags[1] * n2 * c2 * pTinte_Qn2_C_psi
                              + conjFlags[2] * n3 * c3 * pTinte_Qn3_A_psi)
                    )
                    temp_imag = (
                        pTinte_Qn1_A ** c1 * pTinte_Qn2_B ** c2 * pTinte_Qn3_C ** c3
                        * sin(conjFlags[0] * n1 * c1 * pTinte_Qn1_A_psi
                              + conjFlags[1] * n2 * c2 * pTinte_Qn2_B_psi
                              + conjFlags[2] * n3 * c3 * pTinte_Qn3_C_psi)
                        + pTinte_Qn1_B ** c1 * pTinte_Qn2_A ** c2 * pTinte_Qn3_C ** c3
                        * sin(conjFlags[0] * n1 * c1 * pTinte_Qn1_B_psi
                              + conjFlags[1] * n2 * c2 * pTinte_Qn2_A_psi
                              + conjFlags[2] * n3 * c3 * pTinte_Qn3_C_psi)
                        + pTinte_Qn1_C ** c1 * pTinte_Qn2_A ** c2 * pTinte_Qn3_B ** c3
                        * sin(conjFlags[0] * n1 * c1 * pTinte_Qn1_C_psi
                              + conjFlags[1] * n2 * c2 * pTinte_Qn2_A_psi
                              + conjFlags[2] * n3 * c3 * pTinte_Qn3_B_psi)
                        + pTinte_Qn1_C ** c1 * pTinte_Qn2_B ** c2 * pTinte_Qn3_A ** c3
                        * sin(conjFlags[0] * n1 * c1 * pTinte_Qn1_C_psi
                              + conjFlags[1] * n2 * c2 * pTinte_Qn2_B_psi
                              + conjFlags[2] * n3 * c3 * pTinte_Qn3_A_psi)
                        + pTinte_Qn1_A ** c1 * pTinte_Qn2_C ** c2 * pTinte_Qn3_B ** c3
                        * sin(conjFlags[0] * n1 * c1 * pTinte_Qn1_A_psi
                              + conjFlags[1] * n2 * c2 * pTinte_Qn2_C_psi
                              + conjFlags[2] * n3 * c3 * pTinte_Qn3_B_psi)
                        + pTinte_Qn1_B ** c1 * pTinte_Qn2_C ** c2 * pTinte_Qn3_A ** c3
                        * sin(conjFlags[0] * n1 * c1 * pTinte_Qn1_B_psi
                              + conjFlags[1] * n2 * c2 * pTinte_Qn2_C_psi
                              + conjFlags[2] * n3 * c3 * pTinte_Qn3_A_psi)
                    )
                    resolutionFactor_1_AB += (
                        pTinte_Qn1_A ** c1 * pTinte_Qn1_B ** c1
                        * cos(n1 * c1 * (pTinte_Qn1_A_psi - pTinte_Qn1_B_psi)))
                    resolutionFactor_1_CD += (
                        pTinte_Qn1_C ** c1 * pTinte_Qn1_D ** c1
                        * cos(n1 * c1 * (pTinte_Qn1_C_psi - pTinte_Qn1_D_psi)))
                    resolutionFactor_2_AB += (
                        pTinte_Qn2_A ** c2 * pTinte_Qn2_B ** c2
                        * cos(n2 * c2 * (pTinte_Qn2_A_psi - pTinte_Qn2_B_psi)))
                    resolutionFactor_2_CD += (
                        pTinte_Qn2_C ** c2 * pTinte_Qn2_D ** c2
                        * cos(n2 * c2 * (pTinte_Qn2_C_psi - pTinte_Qn2_D_psi)))
                    resolutionFactor_3_AB += (
                        pTinte_Qn3_A ** c3 * pTinte_Qn3_B ** c3
                        * cos(n3 * c3 * (pTinte_Qn3_A_psi - pTinte_Qn3_B_psi)))
                    resolutionFactor_3_CD += (
                        pTinte_Qn3_C ** c3 * pTinte_Qn3_D ** c3
                        * cos(n3 * c3 * (pTinte_Qn3_C_psi - pTinte_Qn3_D_psi)))

                rn_real += temp_real
                rn_real_err += temp_real ** 2.
                rn_imag += temp_imag
                rn_imag_err += temp_imag ** 2.

        resolutionFactor_1_AB = sqrt(resolutionFactor_1_AB / nev)
        resolutionFactor_1_CD = sqrt(resolutionFactor_1_CD / nev)
        resolutionFactor_2_AB = sqrt(resolutionFactor_2_AB / nev)
        resolutionFactor_2_CD = sqrt(resolutionFactor_2_CD / nev)
        resolutionFactor_3_AB = sqrt(resolutionFactor_3_AB / nev)
        resolutionFactor_3_CD = sqrt(resolutionFactor_3_CD / nev)

        total_resolutionFactor = (
            2. * resolutionFactor_1_AB * resolutionFactor_2_AB
            * resolutionFactor_3_CD
            + 2. * resolutionFactor_1_CD * resolutionFactor_2_AB
            * resolutionFactor_3_AB
            + 2. * resolutionFactor_1_AB * resolutionFactor_2_CD
            * resolutionFactor_3_AB
        )

        rn_real = rn_real / nev
        rn_imag = rn_imag / nev
        rn_real_err = sqrt(rn_real_err / nev - rn_real ** 2) / sqrt(nev - 1)
        rn_imag_err = sqrt(rn_imag_err / nev - rn_imag ** 2) / sqrt(nev - 1)

        rn_avg[0] = rn_real / total_resolutionFactor
        rn_avg[1] = rn_real_err / total_resolutionFactor
        rn_avg[2] = rn_imag / total_resolutionFactor
        rn_avg[3] = rn_imag_err / total_resolutionFactor

        return rn_avg


def printHelpMessageandQuit():
    print "Usage : "
    print "AnalyzedDataReader.py databaseName"
    exit(0)


if __name__ == "__main__":
    if len(argv) < 2:
        printHelpMessageandQuit()
    test = AnalyzedDataReader(str(argv[1]))
    # print(test.get_ptinte_two_flow_correlation('pion_p', 'event_plane', 2, -2))
    #print(test.get_ptinte_two_flow_correlation('pion_p', 'scalar_product', 2, -2))
    #print(test.get_ptinte_three_flow_correlation('pion_p', 'event_plane', 
    #                                             2, 3, -5))
    #print(test.get_ptinte_three_flow_correlation('pion_p', 'scalar_product',
    #                                             2, 3, -5))
    #print(test.get_ptinte_two_flow_correlation_sp('pion_p', 2, -2))
    #print(test.get_avg_diffvn_flow('pion_p', 2, 
    #    pT_range = linspace(0.0, 2.0, 21)))
    #print(test.get_avg_intevn_flow('pion_p', 2, pT_range = (0.3, 3.0)))
    # print(test.get_diffvn_flow('pion_p', 'event_plane', 2,
    #                            pT_range=linspace(0.0, 2.0, 21)))
    # print(test.get_diffvn_flow('pion_p', 'scalar_product', 2,
    #                            pT_range=linspace(0.0, 2.0, 21)))
    # print(
    #     test.get_intevn_flow('pion_p', 'event_plane', 2, pT_range=(0.3, 3.0)))
    # print(
    #     test.get_intevn_flow('pion_p', 'scalar_product', 2,
    #                          pT_range=(0.3, 3.0)))
    # charged_sum = 0
    # for aParticle in test.charged_hadron_list:
    #     a, b, c = test.get_particle_yield(aParticle, rap_type = 'rapidity', rap_range=(-0.5, 0.5))
    #     charged_sum += b
    # print "all charged hadron number: ", charged_sum
    #print(test.get_diffvn_2pc_flow('pion_p', 2, 
    #    pT_range = linspace(0.0, 2.0, 21)))
    #print(test.get_intevn_2pc_flow('pion_p', 2, pT_range = (0.3, 3.0)))
    #print(test.get_particle_spectra('pion_p', pT_range=linspace(0.1, 2.5, 20), rap_type = 'pseudorapidity'))
    # get spectra and differential flow
    for aParticle in ['pion_p', 'kaon_p', 'proton']:
        filename = 'data/v2_SP_%s_C10_20.dat'%aParticle
        data = test.get_diffvn_flow(aParticle, 'scalar_product', 2,
                                   pT_range=linspace(0.0, 2.0, 21))
        savetxt(filename, data, fmt='%18.6e', delimiter='\t')

        filename = 'data/dNdydpt_%s_C10_20.dat'%aParticle
        data = test.get_particle_spectra(aParticle, pT_range=linspace(0.1, 2.5, 25), 
                                          rap_type = 'rapidity')
        savetxt(filename, data, fmt='%18.6e', delimiter='\t')

    # print test.get_particle_meanPT('pion_p')
    # print(test.get_particle_yield_vs_rap('pion_p', rap_type = 'rapidity', rap_range=linspace(-2.5, 2.5, 30)))
    #print(test.get_particle_yield('pion_p', rap_type = 'rapidity', rap_range=(-0.5, 0.5)))
    #print(test.get_particle_yield_vs_spatial_variable('pion_p', 'tau', 
    #      linspace(0.6, 10, 50), rap_type = 'rapidity'))
    #print(test.get_avg_diffvn_flow('pion_p', 2, psi_r = 0., 
    #      pT_range = linspace(0.0, 2.0, 21)))
    #print(test.get_avg_intevn_flow('pion_p', 2, psi_r = 0., 
    #      pT_range = (0.3, 3.0)))

