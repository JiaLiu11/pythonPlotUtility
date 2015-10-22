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


class ParticleReader(object):
    """
        This class is used to perform statistical analysis on particle 
        database from UrQMD.
    """

    def __init__(self, database, analyzed_database="analyzed_particles.db"):
        """
            Register databases
        """
        # setup database for analysis
        if isinstance(database, str):
            if path.exists(database):
                database = SqliteDB(database)
            else:
                raise ValueError(
                    "ParticleReader.__init__: input database %s can not be "
                    "found!" % database)
        if isinstance(database, SqliteDB):
            self.db = database
        else:
            raise TypeError(
                "ParticleReader.__init__: the input database must be "
                "a string or a SqliteDB database.")

        if isinstance(analyzed_database, str):
            self.analyzed_db = SqliteDB(analyzed_database)
        else:
            raise ValueError(
                "ParticleReader.__init__: output database %s has to be a "
                "string")

        # setup lookup tables
        self.pid_lookup = dict(self.db.selectFromTable("pid_lookup"))
        self.pid_Mass = dict(self.db.selectFromTable("pid_Mass"))
        
        # define all charged hadrons
        self.charged_hadron_list = [
            "pion_p", "pion_m", "kaon_p", "kaon_m", "proton", "anti_proton",
            "sigma_p", "sigma_m", "anti_sigma_p", "anti_sigma_m",
            "xi_m", "anti_xi_m"]

        self.mergedHaron_name_dict = {
            ('pion_p', 'pion_m')         :       'pion_total',
            ('kaon_p', 'kaon_m')         :       'kaon_total',
            ('proton', 'anti_proton')    :       'proton_total',
            ('lambda', 'anti_lambda')    :       'lambda_total',
            ('xi_m', 'anti_xi_m')        :       'xi_m_total',
            ('omega', 'anti_omega')      :       'omega_total',
        } 

        self.mergedHaron_pid_dict = {
            'pion_total'      :       10010,
            'kaon_total'      :       10011,
            'proton_total'    :       10012,
            'lambda_total'    :       10014,
            'xi_m_total'      :       10013,
            'omega_total'     :       10015,
        }

        # update pid table
        self.pid_lookup.update(self.mergedHaron_pid_dict)  

        # create index for the particle_list table
        if not self.db.doesIndexExist("particleListIndex"):
            print("Create index for particle_list table ...")
            self.db.executeSQLquery(
                "CREATE INDEX particleListIndex ON "
                "particle_list (hydroEvent_id, UrQMDEvent_id, pid)")

        # get number of events
        # pre-collect it to shorten the initial loading time of the database
        if self.db.createTableIfNotExists(
                "number_of_events", (("Nev_tot", "integer"),
                                     ("Nev_hydro", "integer"))):
            self.totNev = self.getNumberOftotalEvents()
            self.hydroNev = self.getNumberOfHydroEvents()
            self.db.insertIntoTable("number_of_events",
                                    (int(self.totNev), int(self.hydroNev)))
            self.db._dbCon.commit()  # commit changes
        else:
            self.totNev = self.db.executeSQLquery(
                "select Nev_tot from number_of_events").fetchall()[0][0]
            self.hydroNev = self.db.executeSQLquery(
                "select Nev_hydro from number_of_events").fetchall()[0][0]

        if self.db.createTableIfNotExists(
                "UrQMD_NevList", (("hydroEventId", "integer"),
                                  ("Number_of_UrQMDevents", "integer"))):
            for hydroEventId in range(1, self.hydroNev + 1):
                UrQMDNev = self.getNumberOfUrQMDEvents(hydroEventId)
                self.db.insertIntoTable("UrQMD_NevList",
                                        (int(hydroEventId), int(UrQMDNev)))
            self.db._dbCon.commit()  # commit changes

        # copy information tables to analyzed_db
        for aTable in ['pid_lookup', 'pid_Mass', 'number_of_events',
                       'UrQMD_NevList']:
            if self.analyzed_db.createTableIfNotExists(
                    aTable, self.db.getTableInfo(aTable)):
                self.analyzed_db.insertIntoTable(
                    aTable, self.db.selectFromTable(aTable))

        # save merged hadron pid to database
        mergedHaron_pidString = " or ".join(map(lambda (x): 'name = \'%s\' '%x, 
                                                self.mergedHaron_pid_dict.keys()))
        self.analyzed_db.executeSQLquery('delete from pid_lookup where %s'%mergedHaron_pidString)
        self.analyzed_db.insertIntoTable("pid_lookup", list(self.mergedHaron_pid_dict.items()))
    ###########################################################################
    # functions to get number of events
    ########################################################################### 
    def getNumberOfHydroEvents(self):
        """
            return total number hydro events stored in the database
        """
        Nev = self.db.executeSQLquery(
            "select count(*) from (select distinct hydroEvent_id "
            "from particle_list)").fetchall()[0][0]
        return (Nev)

    def getNumberOfUrQMDEvents(self, hydroEventid):
        """
            return number of UrQMD events for the given hydro event
        """
        Nev = self.db.executeSQLquery(
            "select count(*) from (select distinct UrQMDEvent_id from "
            "particle_list where hydroEvent_id = %d)"
            % hydroEventid).fetchall()[0][0]
        return (Nev)

    def getNumberOftotalEvents(self):
        """
            return total number of events stored in the database
        """
        hydroEventid = self.db.executeSQLquery(
            "select distinct hydroEvent_id from particle_list").fetchall()
        Nev = 0
        for iev in range(len(hydroEventid)):
            Nev += self.getNumberOfUrQMDEvents(hydroEventid[iev][0])
        return (Nev)

    def getPidString(self, particleName):
        if isinstance(particleName, list):
            pidList = []
            for aPart in particleName:
                pidList.append(str(self.pid_lookup[aPart]))
            pidString = " or ".join(map(lambda (x): 'pid = ' + x, pidList))
        else:
            pid = self.pid_lookup[particleName]
            if pid == 1:  # all charged hadrons
                pidString = self.getPidString(self.charged_hadron_list)
            else:
                pidString = "pid = %d" % pid
        return (pidString)

    ###########################################################################
    # functions to collect particle spectra and yields
    ########################################################################### 
    def get_particle_spectra_hist(self, hydro_id, urqmd_id, pid_string,
                                  rap_type='rapidity', rap_range=(-0.5, 0.5)):
        """
            return pT binned results of particle spectra as a numpy 2-D array
            (pT, dN/dy dpT) or (pT, dN/deta dpT)
            for one given event
        """
        #set pT bin boundaries
        npT = 30
        pT_boundaries = linspace(0, 3, npT + 1)
        dpT = pT_boundaries[1] - pT_boundaries[0]
        pT_avg = (pT_boundaries[0:-1] + pT_boundaries[1:]) / 2.
        dNdydpT = zeros(npT)

        #fetch data from the database
        data = array(self.db.executeSQLquery(
            "select pT from particle_list where hydroEvent_id = %d and "
            "UrQMDEvent_id = %d and (%s) and (%g <= %s and %s <= %g)"
            % (hydro_id, urqmd_id, pid_string, rap_range[0], rap_type,
               rap_type, rap_range[1])).fetchall())
        #bin data
        if data.size != 0:
            for ipT in range(len(pT_boundaries) - 1):
                pT_low = pT_boundaries[ipT]
                pT_high = pT_boundaries[ipT + 1]
                temp_data = data[data[:, 0] < pT_high]
                if temp_data.size != 0:
                    temp_data = temp_data[temp_data[:, 0] >= pT_low]
                    if temp_data.size != 0:
                        pT_avg[ipT] = mean(temp_data)
                        dNdydpT[ipT] = len(temp_data) / dpT

        return array([pT_avg, dNdydpT]).transpose()

    def collect_particle_spectra(
            self, particle_name="pion_p", rap_type='rapidity'):
        """
            collect histogram of particle pT-spectra
            (pT, dN/(dydpT) or (pT, dN/(detadpT))
            for each event and store them into analyzed database
        """
        # get pid string
        pid = self.pid_lookup[particle_name]
        pidString = self.getPidString(particle_name)
        if rap_type == 'rapidity':
            analyzed_table_name = 'particle_pT_spectra'
        elif rap_type == 'pseudorapidity':
            analyzed_table_name = 'particle_pT_spectra_eta'
        else:
            raise TypeError("ParticleReader.collect_particle_spectra: "
                            "invalid input rap_type : %s" % rap_type)

        # check whether the data are already collected
        collected_flag = True
        if self.analyzed_db.createTableIfNotExists(analyzed_table_name,
                                                   (('hydro_event_id',
                                                     'integer'), (
                                                            'urqmd_event_id',
                                                            'integer'),
                                                    ('pid', 'integer'),
                                                    ('pT', 'real'),
                                                    ('dN_dydpT', 'real'))):
            collected_flag = False
        else:
            try_data = array(self.analyzed_db.executeSQLquery(
                "select pT, dN_dydpT from %s where "
                "hydro_event_id = %d and urqmd_event_id = %d and "
                "pid = %d" % (analyzed_table_name, 1, 1, pid)).fetchall())
            if try_data.size == 0: collected_flag = False

        # check whether user wants to update the analyzed data
        if collected_flag:
            print("particle spectra of %s has already been collected!"
                  % particle_name)
            inputval = raw_input(
                "Do you want to delete the existing one and collect again?")
            if inputval.lower() == 'y' or inputval.lower() == 'yes':
                self.analyzed_db.executeSQLquery("delete from %s "
                                                 "where pid = %d" % (
                                                     analyzed_table_name, pid))
                self.analyzed_db._dbCon.commit()  # commit changes
                collected_flag = False

        # collect data loop over all the events
        if not collected_flag:
            print("collect particle spectra of %s ..." % particle_name)
            for hydroId in range(1, self.hydroNev + 1):
                urqmd_nev = self.db.executeSQLquery(
                    "select Number_of_UrQMDevents from UrQMD_NevList where "
                    "hydroEventId = %d " % hydroId).fetchall()[0][0]
                for urqmdId in range(1, urqmd_nev + 1):
                    hist = self.get_particle_spectra_hist(hydroId, urqmdId,
                                                          pidString, rap_type)
                    for ipT in range(len(hist[:, 0])):
                        self.analyzed_db.insertIntoTable(
                            analyzed_table_name,
                            (hydroId, urqmdId, pid, hist[ipT, 0], hist[ipT, 1])
                        )
        self.analyzed_db._dbCon.commit()  # commit changes

    def collect_basic_particle_spectra(self):
        """
            collect particle spectra into database for commonly interested 
            hadrons
        """
        basic_particle_list = ['pion_p', 'kaon_p', 'proton']
        for aParticle in basic_particle_list:
            self.collect_particle_spectra(aParticle, rap_type='rapidity')
            self.collect_particle_spectra(aParticle, rap_type='pseudorapidity')

    def collect_charged_particle_spectra(self):
        """
            collect particle spectra into database for all charged
            hadrons
        """
        charged_particle_list = self.charged_hadron_list
        for aParticle in charged_particle_list:
            self.collect_particle_spectra(aParticle, rap_type='rapidity')
            self.collect_particle_spectra(aParticle, rap_type='pseudorapidity')

    def get_particle_yield_vs_rap_hist(self, hydro_id, urqmd_id, pid_string,
                                       rap_type='rapidity'):
        """
            return rap binned results of particle yield as a numpy 2-D array
            (y, dN/dy) or (eta, dN/deta) for one given event in the database
        """
        #set rap bin boundaries, debug
        nrap = 91
        rap_min = -4.5
        rap_max = 4.5
        rap_boundaries = linspace(rap_min, rap_max, nrap + 1)
        drap = rap_boundaries[1] - rap_boundaries[0]
        rap_avg = (rap_boundaries[0:-1] + rap_boundaries[1:]) / 2.
        dNdy = zeros(nrap)

        #fetch data from the database
        data = array(self.db.executeSQLquery(
            "select (%s) from particle_list where hydroEvent_id = %d and "
            "UrQMDEvent_id = %d and (%s) and (%g <= %s and %s <= %g)"
            % (rap_type, hydro_id, urqmd_id, pid_string, rap_min, rap_type,
               rap_type, rap_max)).fetchall())

        #bin data
        if data.size != 0:
            for irap in range(len(rap_boundaries) - 1):
                rap_low = rap_boundaries[irap]
                rap_high = rap_boundaries[irap + 1]
                temp_data = data[data[:, 0] < rap_high]
                if temp_data.size != 0:
                    temp_data = temp_data[temp_data[:, 0] >= rap_low]
                    if temp_data.size != 0:
                        rap_avg[irap] = mean(temp_data)
                        dNdy[irap] = len(temp_data) / drap

        return array([rap_avg, dNdy]).transpose()

    def collect_particle_yield_vs_rap(
            self, particle_name='pion_p', rap_type='rapidity'):
        """
            collect particle yield as a function of rapidity
            (y, dN/dy) or (eta, dN/deta)
        """
        # get pid string
        pid = self.pid_lookup[particle_name]
        pidString = self.getPidString(particle_name)
        if rap_type == 'rapidity':
            analyzed_table_name = 'particle_yield_vs_rap'
        elif rap_type == 'pseudorapidity':
            analyzed_table_name = 'particle_yield_vs_psedurap'
        else:
            raise TypeError("ParticleReader.collect_particle_yield_vs_rap: "
                            "invalid input rap_type : %s" % rap_type)
        # check whether the data are already collected
        collected_flag = True
        if self.analyzed_db.createTableIfNotExists(analyzed_table_name,
                                                   (('hydro_event_id',
                                                     'integer'), (
                                                            'urqmd_event_id',
                                                            'integer'),
                                                    ('pid', 'integer'),
                                                    ('rap', 'real'),
                                                    ('dN_drap', 'real'))):
            collected_flag = False
        else:
            try_data = array(self.analyzed_db.executeSQLquery(
                "select rap, dN_drap from %s where "
                "hydro_event_id = %d and urqmd_event_id = %d and "
                "pid = %d" % (analyzed_table_name, 1, 1, pid)).fetchall())
            if try_data.size == 0: collected_flag = False

        # check whether user wants to update the analyzed data
        if collected_flag:
            print("%s dependence of particle yield for %s has already been "
                  "collected!" % (rap_type, particle_name))
            inputval = raw_input(
                "Do you want to delete the existing one and collect again?")
            if inputval.lower() == 'y' or inputval.lower() == 'yes':
                self.analyzed_db.executeSQLquery("delete from %s "
                                                 "where pid = %d" % (
                                                     analyzed_table_name, pid))
                self.analyzed_db._dbCon.commit()  # commit changes
                collected_flag = False

        # collect data loop over all the events
        if not collected_flag:
            print("collect %s dependence of particle yield for %s ..."
                  % (rap_type, particle_name))
            for hydroId in range(1, self.hydroNev + 1):
                urqmd_nev = self.db.executeSQLquery(
                    "select Number_of_UrQMDevents from UrQMD_NevList where "
                    "hydroEventId = %d " % hydroId).fetchall()[0][0]
                for urqmdId in range(1, urqmd_nev + 1):
                    hist = self.get_particle_yield_vs_rap_hist(
                        hydroId, urqmdId, pidString, rap_type)
                    for irap in range(len(hist[:, 0])):
                        self.analyzed_db.insertIntoTable(
                            analyzed_table_name, (hydroId, urqmdId, pid,
                                                  hist[irap, 0], hist[irap, 1])
                        )
        self.analyzed_db._dbCon.commit()  # commit changes


##################################################################
    def twoBodyDecay(self, pa, ma, mb, mc=0):
        """
            two body decay: a-->b+c(gamma)
            input: 4-momentum of particle a: pa
            output: two numpy arrays: pb and pc in the lab frame
            return: numpy list contains pb list and pc list
        """
        # in the local rest frame
        p0b = (ma**2. + mb**2. - mc**2) / (2.0*ma)
        p0c = (ma**2. - mb**2. + mc**2) / (2.0*ma)
        pb_lab, pc_lab = zeros((4)),zeros((4))
        if p0c<0:
            print "Particle c with negative p_0:"
            print "p0c = %f"%p0c
            return pb_lab, pc_lab
        # isotropic decay
        theta= arccos(random.uniform(-1, 1))
        phi  = random.uniform(0, 2*pi)
        p_abs = sqrt(p0b**2.-mb**2.)
        pb = array([p0b, p_abs*sin(theta)*cos(phi), p_abs*sin(theta)*sin(phi), 
                        p_abs*cos(theta)])
        pc = array([p0c, -p_abs*sin(theta)*cos(phi), -p_abs*sin(theta)*sin(phi), 
                        -p_abs*cos(theta)])
        # boost to the lab frame
        beta = pa[1:]/pa[0]
        gamma= 1./sqrt(1-dot(beta, beta))
        pb_lab[0] = gamma*(dot(beta,pb[1:])+pb[0])
        pb_lab[1:]= pb[1:]+gamma*beta*(gamma/(gamma+1.)*dot(beta,pb[1:])+p0b)
        pc_lab[0] = gamma*(dot(beta,pc[1:])+pc[0])
        pc_lab[1:]= pc[1:]+gamma*beta*(gamma/(gamma+1.)*dot(beta,pc[1:])+p0b)
        return pb_lab, pc_lab


    def twoBodayDecayMany(self, pa_matrix, ma, mb, mc=0):
        """
            vectorized two body decay for process:
                a-->b+gamma
            input: pa_matrix: n*4 matrix, one line for a particle
                   mb: mass of outgoing particle b
            output: numpy arrays pb_lab_matrix, pc_lab_matrix
        """
        if pa_matrix.ndim==1:
            return twoBodyDecay(pa_matrix, mb)
        # in local rest frame
        p0b = (ma**2. + mb**2. - mc**2) / (2.0*ma)
        p0c = (ma**2. - mb**2. + mc**2) / (2.0*ma)
        theta_array = arccos(random.uniform(-1, 1, size=pa_matrix.shape[0]))
        phi_array  =  random.uniform(0, 2*pi, size=pa_matrix.shape[0])     
        # find pb and pc for all particles
        pb_matrix, pc_matrix = zeros(pa_matrix.shape), zeros(pa_matrix.shape)
        p_abs = sqrt(p0b**2.-mb**2.)
        pb_matrix[:,0] = p0b
        pb_matrix[:,1] = p_abs*sin(theta_array)*cos(phi_array)
        pb_matrix[:,2] = p_abs*sin(theta_array)*sin(phi_array)
        pb_matrix[:,3] = p_abs*cos(theta_array)
        pc_matrix[:,0] = p0c
        pc_matrix[:,1] = -p_abs*sin(theta_array)*cos(phi_array)
        pc_matrix[:,2] = -p_abs*sin(theta_array)*sin(phi_array)
        pc_matrix[:,3] = -p_abs*cos(theta_array)
        # boost to the lab frame
        pb_lab_matrix, pc_lab_matrix = zeros(pa_matrix.shape), zeros(pa_matrix.shape)
        beta_matrix = pa_matrix[:,1:]/transpose(tile(pa_matrix[:,0],(3,1)))
        gamma_array = 1./sqrt(1.-sum(beta_matrix**2., axis=1))
        gamma_matrix = tile(gamma_array,(3,1)).transpose()
        gamma_factor = tile(gamma_array/(1.+gamma_array), (3,1)).transpose()
        beta_dot_pb = tile(diag(dot(beta_matrix, transpose(pb_matrix[:,1:]))),(3,1)).transpose()
        pb_lab_matrix[:,0]=gamma_array*(beta_dot_pb+p0b)[:,0]
        pb_lab_matrix[:,1:]=(pb_matrix[:,1:]+
            gamma_matrix*beta_matrix*(gamma_factor*beta_dot_pb+p0b))
        beta_dot_pc = tile(diag(dot(beta_matrix, transpose(pc_matrix[:,1:]))),(3,1)).transpose()  
        pc_lab_matrix[:,0]=gamma_array*(beta_dot_pc+p0c)[:,0]
        pc_lab_matrix[:,1:]=(pc_matrix[:,1:]
                        +gamma_matrix*beta_matrix*(gamma_factor*beta_dot_pc+p0c))
        # tempb = sqrt(pb_lab_matrix[:,0]**2-sum(pb_lab_matrix[:,1:]**2,axis=1)+1e-10)
        # tempc = sqrt(pc_lab_matrix[:,0]**2-sum(pc_lab_matrix[:,1:]**2,axis=1)+1e-10)
        # violation_b = where(abs(tempb-mb)>1e-5)
        # violation_c = where(abs(tempc-mc)>1e-5)
        # print "sanity check:"
        # print "pb violations: %d/%d"%(len(violation_b[0]), pb_lab_matrix.shape[0])
        # print violation_b
        # print tempb[violation_b[0]]
        # print "pc violations: %d/%d"%(len(violation_c[0]), pc_lab_matrix.shape[0])
        # print violation_c
        # print pb_lab_matrix[violation_b[0],:]
        # print pc_lab_matrix[violation_c[0],:]
        # if len(violation_b[0])>0 or len(violation_c[0])>0:
        #     exit(-1)
        return pb_lab_matrix, pc_lab_matrix

    def to4momentum_format_converter(self, mass, p_db):
        ''' convert the momentum table directly extracted from DB to standard 4 momentum table'''
        result = zeros(p_db.shape) # p0, px, py, pz
        try:
            # p_db: pT, phi_p, rapidity, pseudorapidity 
            mt = sqrt(mass**2.+p_db[:,0]**2)
            result[:,0] = mt*cosh(p_db[:,2])
            result[:,1] = p_db[:,0]*cos(p_db[:,1])
            result[:,2] = p_db[:,0]*sin(p_db[:,1])
            result[:,3] = mt*sinh(p_db[:,2])
        except:
            print "to4momentum_format_converter: conversion failed!"
        # test
        temp = sqrt(result[:,0]**2-sum(result[:,1:]**2, axis=1))
        violation = where(abs(temp-mass)>1e-6)
        if violation[0].size>0:
            print "to 4 momentum conversion:"
            print violation[0].shape
            print violation[0]
            print temp[violation[0]]
            exit(-1)
        return result

    def toDBmomentum_format_converter(self, p_4form):
        ''' convert the momentum table from standard 4-momentum form to particle_list form'''
        result = zeros(p_4form.shape)
        try:
            result[:,0] = sqrt(p_4form[:,1]**2.+p_4form[:,2]**2)
            result[:,1] = arctan2(p_4form[:,2], p_4form[:,1])
            result[:,2] = 0.5*log((p_4form[:,0]+p_4form[:,3])/(p_4form[:,0]-p_4form[:,3]))
            p_abs = sqrt(sum(p_4form[:,1:]**2., axis=1))
            result[:,3] = 0.5*log((p_abs+p_4form[:,3])/(p_abs-p_4form[:,3]))
        except:
            print "toDBmomentum_format_converter: conversion failed!"
        return result            

    def particle_decay(self, source_particle_name, 
            product_particle_name_1, product_particle_name_2):
        """
            Implement partice decay manually, and write the output to database
        """
        print "Manually decay: %s ---> %s + %s"%(source_particle_name, 
            product_particle_name_1, product_particle_name_2)
        source_pid = self.pid_lookup[source_particle_name]
        product1_pid = self.pid_lookup[product_particle_name_1]
        product2_pid = self.pid_lookup[product_particle_name_2]
        source_mass= self.pid_Mass[source_particle_name]
        product1_mass= self.pid_Mass[product_particle_name_1]
        product2_mass= self.pid_Mass[product_particle_name_2]

        # hydroEvent_id, UrQMDEvent_id, pid, tau, x, y
        #                 eta, pT, phi_p, rapidity, pseudorapidity        
        source_data = array(self.db.executeSQLquery(
            "select * from particle_list "
            "where pid=%d and hydroEvent_id=1"%source_pid).fetchone())
        if source_data.size == 0:
            print "particle_decay: no source particle!"
            return

        # backup up source and target particles
        self.db.createTableIfNotExists("particle_list_backup", 
               (("hydroEvent_id","integer"), ("UrQMDEvent_id","interger"), ("pid","integer"), 
                ("tau","real"), ("x","real"), ("y","real"), ("eta","real"), 
                ("pT", "real"), ("phi_p", "real"), ("rapidity", "real"), ("pseudorapidity", "real")) )
        for hydroId in range(1, self.hydroNev + 1):
            print "processing hydro event %d:"%hydroId
            print "backup tables..."
            for iPid in [source_pid, product1_pid, product2_pid]:
                backup_data_cursor = self.db.executeSQLquery(
                                            "select * from particle_list "
                                            "where pid=%d and hydroEvent_id=%d"%(iPid, hydroId))
                while True:
                    product_data_tobackup = array(backup_data_cursor.fetchmany(5000))
                    if product_data_tobackup.size == 0:
                        break
                    try:
                        self.db.insertIntoTable("particle_list_backup", list(product_data_tobackup))
                    except:
                        print "particle_decay: backup particle pid=%d data failed"%iPid
                        exit(-1)

            # decay
            print "excute decay..."
            source_data_cursor = self.db.executeSQLquery(
                    "select * from particle_list "
                    "where pid=%d and hydroEvent_id=%d"%(source_pid, hydroId))
            while True:
                source_data = array(source_data_cursor.fetchmany(5000))
                if source_data.size == 0:
                    break
                # extract momentum
                source_momentum_data = source_data[:, 7:]
                source_4momentum_data = self.to4momentum_format_converter(
                    source_mass, source_momentum_data)
                product1_4momentum, product2_4momentum = self.twoBodayDecayMany(
                    source_4momentum_data, source_mass, product1_mass, product2_mass)
                try:
                    # backup particle source
                    # save product particle 1
                    product1_momentum_dbFormat =  self.toDBmomentum_format_converter(product1_4momentum)
                    product_1_data = source_data.copy()
                    product_1_data[:, 2] = product1_pid
                    product_1_data[:, 7:]= product1_momentum_dbFormat
                    self.db.insertIntoTable("particle_list", list(product_1_data))
                    # save product particle 2
                    product2_momentum_dbFormat =  self.toDBmomentum_format_converter(product2_4momentum)
                    product_2_data = source_data.copy()
                    product_2_data[:, 2] = product2_pid
                    product_2_data[:, 7:]= product2_momentum_dbFormat
                    self.db.insertIntoTable("particle_list", list(product_2_data))
                except:
                    print "particle_decay: saving data failed!"
                    exit(-1)
            # delete particle source
            self.db.executeSQLquery("delete from particle_list "
                                     "where pid = %d and hydroEvent_id=%d" %(source_pid, hydroId))
            print "hydro event %d/%d finished!"%(hydroId, self.hydroNev)
            self.db._dbCon.commit()
        self.db.closeConnection()
        print "Decay finished!"

##################################################################


    def collect_basic_particle_yield(self):
        """
            collect particle yield into database for commonly interested 
            hadrons
        """
        basic_particle_list = ['pion_p', 'kaon_p', 'proton']
        for aParticle in basic_particle_list:
            self.collect_particle_yield_vs_rap(aParticle, rap_type='rapidity')
            self.collect_particle_yield_vs_rap(aParticle,
                                               rap_type='pseudorapidity')

    def collect_charged_particle_yield(self):
        """
            collect particle yield into database for all charged 
            hadrons
        """
        charged_particle_list = self.charged_hadron_list
        for aParticle in charged_particle_list:
            self.collect_particle_yield_vs_rap(aParticle, rap_type='rapidity')
            self.collect_particle_yield_vs_rap(aParticle,
                                               rap_type='pseudorapidity')

    def collect_flow_Qn_vectors_for_mergedHaron(self):
        """
            collect Qn vectors into database for hadron + anti-hadron
        """
        for particle_a, particle_b in self.mergedHaron_name_dict:
            self.collect_flow_Qn_vectors_forTwo(particle_a, particle_b)

    ###########################################################################
    # functions to collect particle emission function
    ########################################################################### 
    def get_particle_yield_vs_sv_hist(self, hydro_id, urqmd_id, pid_string,
                                      sv_type, sv_boundaries, rap_type,
                                      rap_range):
        """
            return [sv_type, dN/dsv_type] as a numpy 2D-array for one given 
            event in the database
        """
        #set sv_type bin boundaries
        nsv = len(sv_boundaries) - 1
        dsv = sv_boundaries[1] - sv_boundaries[0]
        sv_avg = (sv_boundaries[0:-1] + sv_boundaries[1:]) / 2.
        dNdsv = zeros(nsv)

        #fetch data from the database
        data = array(self.db.executeSQLquery(
            "select %s from particle_list where hydroEvent_id = %d and "
            "UrQMDEvent_id = %d and (%s) and (%g <= %s and %s <= %g)"
            % (sv_type, hydro_id, urqmd_id, pid_string, rap_range[0], rap_type,
               rap_type, rap_range[1])).fetchall())

        #bin data
        if data.size != 0:
            for isv in range(len(sv_boundaries) - 1):
                sv_low = sv_boundaries[isv]
                sv_high = sv_boundaries[isv + 1]
                temp_data = data[data[:, 0] < sv_high]
                if temp_data.size != 0:
                    temp_data = temp_data[temp_data[:, 0] >= sv_low]
                    if temp_data.size != 0:
                        sv_avg[isv] = mean(temp_data)
                        dNdsv[isv] = len(temp_data) / dsv

        return array([sv_avg, dNdsv]).transpose()

    def collect_particle_yield_vs_spatial_variable(
            self, particle_name, sv_type, sv_range, rap_type, rap_range):
        """
            collect particle yield per spacial variable, (tau, x, y, or eta_s) 
            for each event and store them into analyzed database
        """
        # get pid string
        pid = self.pid_lookup[particle_name]
        pid_string = self.getPidString(particle_name)
        if rap_type == 'rapidity':
            analyzed_table_name = 'particle_emission_d%s' % sv_type
        elif rap_type == 'pseudorapidity':
            analyzed_table_name = 'particle_emission_d%s_eta' % sv_type
        else:
            raise ValueError("ParticleReader.collect_particle_yield_vs_"
                             "spatial_variable: invalid input rap_type : %s"
                             % rap_type)

        # check whether the data are already collected
        collected_flag = True
        if self.analyzed_db.createTableIfNotExists(analyzed_table_name,
                                                   (('hydro_event_id',
                                                     'integer'), (
                                                            'urqmd_event_id',
                                                            'integer'),
                                                    ('pid', 'integer'),
                                                    ('%s' % sv_type, 'real'),
                                                    ('dN_d%s' % sv_type,
                                                     'real'))):
            collected_flag = False
        else:
            try_data = array(self.analyzed_db.executeSQLquery(
                "select %s, dN_d%s from %s where "
                "hydro_event_id = %d and urqmd_event_id = %d and pid = %d"
                % (sv_type, sv_type, analyzed_table_name, 1, 1, pid)
            ).fetchall())
            if try_data.size == 0: collected_flag = False

        # check whether user wants to update the analyzed data
        if collected_flag:
            print("dN/d%s of %s has already been collected!"
                  % (sv_type, particle_name))
            inputval = raw_input(
                "Do you want to delete the existing one and collect again?")
            if inputval.lower() == 'y' or inputval.lower() == 'yes':
                self.analyzed_db.executeSQLquery("delete from %s "
                                                 "where pid = %d" % (
                                                     analyzed_table_name, pid))
                self.analyzed_db._dbCon.commit()  # commit changes
                collected_flag = False

        # collect data loop over all the events
        if not collected_flag:
            print("collect particle yield as a function of %s for %s"
                  % (sv_type, particle_name))
            for hydroId in range(1, self.hydroNev + 1):
                urqmd_nev = self.db.executeSQLquery(
                    "select Number_of_UrQMDevents from UrQMD_NevList where "
                    "hydroEventId = %d " % hydroId).fetchall()[0][0]
                for urqmdId in range(1, urqmd_nev + 1):
                    hist = self.get_particle_yield_vs_sv_hist(
                        hydroId, urqmdId, pid_string, sv_type, sv_range,
                        rap_type, rap_range)
                    for isv in range(len(hist[:, 0])):
                        self.analyzed_db.insertIntoTable(
                            analyzed_table_name, (hydroId, urqmdId, pid,
                                                  hist[isv, 0], hist[isv, 1])
                        )
        self.analyzed_db._dbCon.commit()  # commit changes


    ###########################################################################
    # functions to collect particle anisotropic flows
    ########################################################################### 
    def get_Qn_vector(self, hydro_id, urqmd_id, pid_string,
                      weight_type, rap_type):
        """
            return Qn_data, Qn_pTdata for partitle with pid_string with 
            weight_type from one given event
            Qn_data = (n, Nparticle, Qn_real, Qn_imag, Nparticle_sub, 
                       QnA_real, QnA_imag, QnB_real, QnB_imag,
                       QnC_real, QnC_imag, QnD_real, QnD_imag)
            Qn_pTdata = (n, pT, Nparticle, Qn_real, Qn_imag, Nparticle_sub, 
                         QnA_real, QnA_imag, QnB_real, QnB_imag, 
                         QnC_real, QnC_imag, QnD_real, QnD_imag)
            Qn is taken particles havging -0.3 <= rap <= 0.3
            QnA is taken particles having -0.6 <= rap < -0.3
            QnB is taken particles having  0.3 <  rap <= 0.6
            QnC is taken particles having -1.0 <= rap < -0.6
            QnD is taken particles having  0.6 <  rap <= 1.0
            Nparticle_sub = min(len(QnA), len(QnB), len(QnC), len(QnD))
        """
        rap_gap = (0.3, 0.6, 1.0)
        eps = 1e-15
        norder = 6
        npT = 30
        pT_boundaries = linspace(0.0, 3.0, npT + 1)
        dpT = pT_boundaries[1] - pT_boundaries[0]
        Qn_data = zeros([norder, 13])
        Qn_pTdata = zeros([norder * npT, 14])
        for iorder in range(norder):
            Qn_data[iorder, 0] = iorder + 1
            for ipT in range(npT):
                Qn_pTdata[iorder * npT + ipT, 0] = iorder + 1
                Qn_pTdata[iorder * npT:(iorder + 1) * npT, 1] = (
                    (pT_boundaries[0:npT] + pT_boundaries[1:npT + 1]) / 2.)

        print("processing event: (%d, %d) " % (hydro_id, urqmd_id))
        particleList = array(self.db.executeSQLquery(
            "select pT, phi_p, %s from particle_list where "
            "hydroEvent_id = %d and (%s)"
            % (rap_type, hydro_id, pid_string)).fetchall())

        # no particle in the event
        if particleList.size == 0: return (Qn_data, Qn_pTdata)

        pT = particleList[:, 0]
        phi = particleList[:, 1]
        rap = particleList[:, 2]
        if weight_type == 'pT':
            weight = pT
        elif weight_type == '1':
            weight = ones(len(pT))

        # bin particle samples
        idx = []
        idxA = []
        idxB = []
        idxC = []
        idxD = []
        idx_pT = [[] for _ in range(npT)]
        idxA_pT = [[] for _ in range(npT)]
        idxB_pT = [[] for _ in range(npT)]
        idxC_pT = [[] for _ in range(npT)]
        idxD_pT = [[] for _ in range(npT)]
        for ipart in range(len(pT)):
            pTpos = int((pT[ipart] - pT_boundaries[0]) / dpT)
            # Qn is taken particles havging -0.3 <= rap <= 0.3
            if rap[ipart] <= rap_gap[0] and rap[ipart] >= - rap_gap[0]:
                idx.append(ipart)
                if pTpos < npT: idx_pT[pTpos].append(ipart)
            # QnA is taken particles having -0.6 <= rap < -0.3
            elif rap[ipart] < - rap_gap[0] and rap[ipart] >= - rap_gap[1]:
                idxA.append(ipart)
                if pTpos < npT: idxA_pT[pTpos].append(ipart)
            # QnB is taken particles having 0.3 < rap <= 0.6
            elif rap[ipart] <= rap_gap[1] and rap[ipart] > rap_gap[0]:
                idxB.append(ipart)
                if pTpos < npT: idxB_pT[pTpos].append(ipart)
            # QnC is taken particles having -1.0 <= rap < -0.6
            elif rap[ipart] < - rap_gap[1] and rap[ipart] >= - rap_gap[2]:
                idxC.append(ipart)
                if pTpos < npT: idxC_pT[pTpos].append(ipart)
            # QnD is taken particles having 0.6 < rap <= 1.0
            elif rap[ipart] <= rap_gap[2] and rap[ipart] > rap_gap[1]:
                idxD.append(ipart)
                if pTpos < npT: idxD_pT[pTpos].append(ipart)

        # calculate Qn vectors
        Nparticle = len(idx)
        Nparticle_sub = min(len(idxA), len(idxB), len(idxC), len(idxD))
        for iorder in range(1, norder + 1):
            # Qn vectors at mid rapidity
            temp_Qn_x = sum(weight[idx] * cos(iorder * phi[idx]))
            temp_Qn_y = sum(weight[idx] * sin(iorder * phi[idx]))
            Qn_data[iorder - 1, 1] = Nparticle
            Qn_data[iorder - 1, 2] = temp_Qn_x / (Nparticle + eps)
            Qn_data[iorder - 1, 3] = temp_Qn_y / (Nparticle + eps)

            # sub event Qn vectors
            Qn_data[iorder - 1, 4] = Nparticle_sub
            # QnA vectors at (-1.5, -0.5)
            temp_Qn_x = sum(weight[idxA[0:Nparticle_sub]]
                            * cos(iorder * phi[idxA[0:Nparticle_sub]]))
            temp_Qn_y = sum(weight[idxA[0:Nparticle_sub]]
                            * sin(iorder * phi[idxA[0:Nparticle_sub]]))
            Qn_data[iorder - 1, 5] = temp_Qn_x / (Nparticle_sub + eps)
            Qn_data[iorder - 1, 6] = temp_Qn_y / (Nparticle_sub + eps)
            # QnB vector at (0.5, 1.5)
            temp_Qn_x = sum(weight[idxB[0:Nparticle_sub]]
                            * cos(iorder * phi[idxB[0:Nparticle_sub]]))
            temp_Qn_y = sum(weight[idxB[0:Nparticle_sub]]
                            * sin(iorder * phi[idxB[0:Nparticle_sub]]))
            Qn_data[iorder - 1, 7] = temp_Qn_x / (Nparticle_sub + eps)
            Qn_data[iorder - 1, 8] = temp_Qn_y / (Nparticle_sub + eps)
            # QnC vector at (-2.5, -1.5)
            temp_Qn_x = sum(weight[idxC[0:Nparticle_sub]]
                            * cos(iorder * phi[idxC[0:Nparticle_sub]]))
            temp_Qn_y = sum(weight[idxC[0:Nparticle_sub]]
                            * sin(iorder * phi[idxC[0:Nparticle_sub]]))
            Qn_data[iorder - 1, 9] = temp_Qn_x / (Nparticle_sub + eps)
            Qn_data[iorder - 1, 10] = temp_Qn_y / (Nparticle_sub + eps)
            # QnD vector at (1.5, 2.5)
            temp_Qn_x = sum(weight[idxD[0:Nparticle_sub]]
                            * cos(iorder * phi[idxD[0:Nparticle_sub]]))
            temp_Qn_y = sum(weight[idxD[0:Nparticle_sub]]
                            * sin(iorder * phi[idxD[0:Nparticle_sub]]))
            Qn_data[iorder - 1, 11] = temp_Qn_x / (Nparticle_sub + eps)
            Qn_data[iorder - 1, 12] = temp_Qn_y / (Nparticle_sub + eps)

            # pT differential Qn vectors
            for ipT in range(npT):
                data_idx = (iorder - 1) * npT + ipT
                # pT differential Qn vectors at mid rapidity
                if idx_pT[ipT] != []:
                    Nparticle_pT = len(idx_pT[ipT])
                    Qn_pTdata[data_idx, 1] = mean(pT[idx_pT[ipT]])
                    temp_Qn_x = sum(
                        weight[idx_pT[ipT]] * cos(iorder * phi[idx_pT[ipT]]))
                    temp_Qn_y = sum(
                        weight[idx_pT[ipT]] * sin(iorder * phi[idx_pT[ipT]]))
                    Qn_pTdata[data_idx, 2] = Nparticle_pT
                    Qn_pTdata[data_idx, 3] = temp_Qn_x / (Nparticle_pT + eps)
                    Qn_pTdata[data_idx, 4] = temp_Qn_y / (Nparticle_pT + eps)

                # pT differential Qn vectors from sub events
                Nparticle_sub_pT = min(len(idxA_pT[ipT]), len(idxB_pT[ipT]),
                                       len(idxC_pT[ipT]), len(idxD_pT[ipT]))
                Qn_pTdata[data_idx, 5] = Nparticle_sub_pT
                if Nparticle_sub_pT == 0: continue
                # pT differential QnA vectors at (-1.5, -0.5)
                temp_Qn_x = sum(
                    weight[idxA_pT[ipT][0:Nparticle_sub_pT]]
                    * cos(iorder * phi[idxA_pT[ipT][0:Nparticle_sub_pT]]))
                temp_Qn_y = sum(
                    weight[idxA_pT[ipT][0:Nparticle_sub_pT]]
                    * sin(iorder * phi[idxA_pT[ipT][0:Nparticle_sub_pT]]))
                Qn_pTdata[data_idx, 6] = temp_Qn_x / (Nparticle_sub_pT + eps)
                Qn_pTdata[data_idx, 7] = temp_Qn_y / (Nparticle_sub_pT + eps)
                # pT differential QnB vector at (0.5, 1.5)
                temp_Qn_x = sum(
                    weight[idxB_pT[ipT][0:Nparticle_sub_pT]]
                    * cos(iorder * phi[idxB_pT[ipT][0:Nparticle_sub_pT]]))
                temp_Qn_y = sum(
                    weight[idxB_pT[ipT][0:Nparticle_sub_pT]]
                    * sin(iorder * phi[idxB_pT[ipT][0:Nparticle_sub_pT]]))
                Qn_pTdata[data_idx, 8] = temp_Qn_x / (Nparticle_sub_pT + eps)
                Qn_pTdata[data_idx, 9] = temp_Qn_y / (Nparticle_sub_pT + eps)
                # pT differential QnC vectors at (-2.5, -1.5)
                temp_Qn_x = sum(
                    weight[idxC_pT[ipT][0:Nparticle_sub_pT]]
                    * cos(iorder * phi[idxC_pT[ipT][0:Nparticle_sub_pT]]))
                temp_Qn_y = sum(
                    weight[idxC_pT[ipT][0:Nparticle_sub_pT]]
                    * sin(iorder * phi[idxC_pT[ipT][0:Nparticle_sub_pT]]))
                Qn_pTdata[data_idx, 10] = temp_Qn_x / (Nparticle_sub_pT + eps)
                Qn_pTdata[data_idx, 11] = temp_Qn_y / (Nparticle_sub_pT + eps)
                # pT differential QnD vector at (1.5, 2.5)
                temp_Qn_x = sum(
                    weight[idxD_pT[ipT][0:Nparticle_sub_pT]]
                    * cos(iorder * phi[idxD_pT[ipT][0:Nparticle_sub_pT]]))
                temp_Qn_y = sum(
                    weight[idxD_pT[ipT][0:Nparticle_sub_pT]]
                    * sin(iorder * phi[idxD_pT[ipT][0:Nparticle_sub_pT]]))
                Qn_pTdata[data_idx, 12] = temp_Qn_x / (Nparticle_sub_pT + eps)
                Qn_pTdata[data_idx, 13] = temp_Qn_y / (Nparticle_sub_pT + eps)

        return (Qn_data, Qn_pTdata)

    def collect_flow_Qn_vectors(self, particle_name):
        """
            collect nth order flow Qn vector and sub-event QnA, QnB, QnC, QnD
            vectors for all the events. n is from 1 to 6
            Qn := 1/Nparticle * sum_i exp[i*n*phi_i]
            Qn is taken particles havging -0.3 <= rap <= 0.3
            QnA is taken particles having -0.6 <= rap < -0.3
            QnB is taken particles having  0.3 <  rap <= 0.6
            QnC is taken particles having -1.0 <= rap < -0.6
            QnD is taken particles having  0.6 <  rap <= 1.0
            (rapidity is used for identified particles 
             and pseudorapidity is used for all charged particles)
        """
        # get pid string
        pid = self.pid_lookup[particle_name]
        pid_string = self.getPidString(particle_name)
        if particle_name == "charged":
            rap_type = 'pseudorapidity'
        else:
            rap_type = 'rapidity'

        weight_type = '1'
        analyzed_table_name = 'flow_Qn_vectors'
        analyzed_table_pTdiff_name = 'flow_Qn_vectors_pTdiff'

        # check whether the data are already collected
        collected_flag = True
        if self.analyzed_db.createTableIfNotExists(
                analyzed_table_name,
                (('hydro_event_id', 'integer'), ('urqmd_event_id', 'integer'),
                 ('pid', 'integer'), ('weight_type', 'text'), ('n', 'integer'),
                 ('Nparticle', 'integer'), ('Qn_real', 'real'),
                 ('Qn_imag', 'real'), ('Nparticle_sub', 'integer'),
                 ('QnA_real', 'real'), ('QnA_imag', 'real'),
                 ('QnB_real', 'real'), ('QnB_imag', 'real'),
                 ('QnC_real', 'real'), ('QnC_imag', 'real'),
                 ('QnD_real', 'real'), ('QnD_imag', 'real'))
        ):
            collected_flag = False
        else:
            try_data = array(self.analyzed_db.executeSQLquery(
                "select Qn_real from %s where "
                "hydro_event_id = %d and urqmd_event_id = %d and pid = %d and "
                "n = 1" % (analyzed_table_name, 1, 1, pid)).fetchall())
            if try_data.size == 0: collected_flag = False

        # check whether user wants to update the analyzed data
        if collected_flag:
            print("flow Qn vectors for %s has already been collected!"
                  % particle_name)
            inputval = raw_input(
                "Do you want to delete the existing one and collect again?")
            if inputval.lower() == 'y' or inputval.lower() == 'yes':
                self.analyzed_db.executeSQLquery("delete from %s "
                                                 "where pid = %d" % (
                                                     analyzed_table_name, pid))
                self.analyzed_db._dbCon.commit()  # commit changes
                collected_flag = False

        # check whether the pT differential data are already collected
        collected_pTdiff_flag = True
        if self.analyzed_db.createTableIfNotExists(
                analyzed_table_pTdiff_name,
                (('hydro_event_id', 'integer'), ('urqmd_event_id', 'integer'),
                 ('pid', 'integer'), ('weight_type', 'text'), ('n', 'integer'),
                 ('pT', 'real'), ('Nparticle', 'integer'), ('Qn_real', 'real'),
                 ('Qn_imag', 'real'), ('Nparticle_sub', 'integer'),
                 ('QnA_real', 'real'), ('QnA_imag', 'real'),
                 ('QnB_real', 'real'), ('QnB_imag', 'real'),
                 ('QnC_real', 'real'), ('QnC_imag', 'real'),
                 ('QnD_real', 'real'), ('QnD_imag', 'real'))
        ):
            collected_pTdiff_flag = False
        else:
            try_data = array(self.analyzed_db.executeSQLquery(
                "select Qn_real from %s where "
                "hydro_event_id = %d and urqmd_event_id = %d and pid = %d and "
                "n = 1" % (analyzed_table_pTdiff_name, 1, 1, pid)).fetchall())
            if try_data.size == 0: collected_pTdiff_flag = False

        # check whether user wants to update the analyzed data
        if collected_pTdiff_flag:
            print("pT differential flow Qn vectors for %s has already been "
                  "collected!" % particle_name)
            inputval = raw_input(
                "Do you want to delete the existing one and collect again?")
            if inputval.lower() == 'y' or inputval.lower() == 'yes':
                self.analyzed_db.executeSQLquery(
                    "delete from %s where pid = %d"
                    % (analyzed_table_pTdiff_name, pid))
                self.analyzed_db._dbCon.commit()  # commit changes
                collected_pTdiff_flag = False

        # collect data loop over all the events
        if not collected_flag or not collected_pTdiff_flag:
            print("collect flow Qn vectors for %s ..." % particle_name)
            urqmdId = 1
            for hydroId in range(1, self.hydroNev + 1):
                Qn_data, Qn_pTdata = self.get_Qn_vector(
                    hydroId, urqmdId, pid_string, weight_type, rap_type)
                if not collected_flag:
                    for item in range(len(Qn_data[:, 0])):
                        self.analyzed_db.insertIntoTable(
                            analyzed_table_name, ((hydroId, urqmdId, pid,
                                                   weight_type) + tuple(
                                Qn_data[item, :]))
                        )
                if not collected_pTdiff_flag:
                    for item in range(len(Qn_pTdata[:, 0])):
                        self.analyzed_db.insertIntoTable(
                            analyzed_table_pTdiff_name,
                            ((hydroId, urqmdId, pid, weight_type)
                             + tuple(Qn_pTdata[item, :]))
                        )
        self.analyzed_db._dbCon.commit()  # commit changes


    def collect_flow_Qn_vectors_forTwo(self, particle_name_a, particle_name_b):
        """
            collect nth order flow Qn vector and sub-event QnA, QnB, QnC, QnD
            vectors for all the events of two hadrons. n is from 1 to 6
            Qn := 1/Nparticle * sum_i exp[i*n*phi_i]
            Qn is taken particles havging -0.5 <= rap <= 0.5
            QnA is taken particles having -1.5 <= rap < -0.5
            QnB is taken particles having 0.5 < rap <= 1.5
            QnC is taken particles having -2.5 <= rap < -1.5
            QnD is taken particles having 1.5 < rap <= 2.5
            (rapidity is used for identified particles 
             and pseudorapidity is used for all charged particles)
        """
        # get pid string
        particle_name = self.mergedHaron_name_dict[(particle_name_a, particle_name_b)]
        pid = self.pid_lookup[particle_name]
        pid_string = self.getPidString([particle_name_a, particle_name_b])
        rap_type = 'rapidity'

        weight_type = '1'
        analyzed_table_name = 'flow_Qn_vectors'
        analyzed_table_pTdiff_name = 'flow_Qn_vectors_pTdiff'

        # check whether the data are already collected
        collected_flag = True
        if self.analyzed_db.createTableIfNotExists(
                analyzed_table_name,
                (('hydro_event_id', 'integer'), ('urqmd_event_id', 'integer'),
                 ('pid', 'integer'), ('weight_type', 'text'), ('n', 'integer'),
                 ('Nparticle', 'integer'), ('Qn_real', 'real'),
                 ('Qn_imag', 'real'), ('Nparticle_sub', 'integer'),
                 ('QnA_real', 'real'), ('QnA_imag', 'real'),
                 ('QnB_real', 'real'), ('QnB_imag', 'real'),
                 ('QnC_real', 'real'), ('QnC_imag', 'real'),
                 ('QnD_real', 'real'), ('QnD_imag', 'real'))
        ):
            collected_flag = False
        else:
            try_data = array(self.analyzed_db.executeSQLquery(
                "select Qn_real from %s where "
                "hydro_event_id = %d and urqmd_event_id = %d and pid = %d and "
                "n = 1" % (analyzed_table_name, 1, 1, pid)).fetchall())
            if try_data.size == 0: collected_flag = False

        # check whether user wants to update the analyzed data
        if collected_flag:
            print("flow Qn vectors for %s has already been collected!"
                  % particle_name)
            inputval = raw_input(
                "Do you want to delete the existing one and collect again?")
            if inputval.lower() == 'y' or inputval.lower() == 'yes':
                self.analyzed_db.executeSQLquery("delete from %s "
                                                 "where pid = %d" % (
                                                     analyzed_table_name, pid))
                self.analyzed_db._dbCon.commit()  # commit changes
                collected_flag = False

        # check whether the pT differential data are already collected
        collected_pTdiff_flag = True
        if self.analyzed_db.createTableIfNotExists(
                analyzed_table_pTdiff_name,
                (('hydro_event_id', 'integer'), ('urqmd_event_id', 'integer'),
                 ('pid', 'integer'), ('weight_type', 'text'), ('n', 'integer'),
                 ('pT', 'real'), ('Nparticle', 'integer'), ('Qn_real', 'real'),
                 ('Qn_imag', 'real'), ('Nparticle_sub', 'integer'),
                 ('QnA_real', 'real'), ('QnA_imag', 'real'),
                 ('QnB_real', 'real'), ('QnB_imag', 'real'),
                 ('QnC_real', 'real'), ('QnC_imag', 'real'),
                 ('QnD_real', 'real'), ('QnD_imag', 'real'))
        ):
            collected_pTdiff_flag = False
        else:
            try_data = array(self.analyzed_db.executeSQLquery(
                "select Qn_real from %s where "
                "hydro_event_id = %d and urqmd_event_id = %d and pid = %d and "
                "n = 1" % (analyzed_table_pTdiff_name, 1, 1, pid)).fetchall())
            if try_data.size == 0: collected_pTdiff_flag = False

        # check whether user wants to update the analyzed data
        if collected_pTdiff_flag:
            print("pT differential flow Qn vectors for %s has already been "
                  "collected!" % particle_name)
            inputval = raw_input(
                "Do you want to delete the existing one and collect again?")
            if inputval.lower() == 'y' or inputval.lower() == 'yes':
                self.analyzed_db.executeSQLquery(
                    "delete from %s where pid = %d"
                    % (analyzed_table_pTdiff_name, pid))
                self.analyzed_db._dbCon.commit()  # commit changes
                collected_pTdiff_flag = False

        # collect data loop over all the events
        if not collected_flag or not collected_pTdiff_flag:
            print("collect flow Qn vectors for %s ..." % particle_name)
            urqmdId = 1
            for hydroId in range(1, self.hydroNev + 1):
                Qn_data, Qn_pTdata = self.get_Qn_vector(
                    hydroId, urqmdId, pid_string, weight_type, rap_type)
                if not collected_flag:
                    for item in range(len(Qn_data[:, 0])):
                        self.analyzed_db.insertIntoTable(
                            analyzed_table_name, ((hydroId, urqmdId, pid,
                                                   weight_type) + tuple(
                                Qn_data[item, :]))
                        )
                if not collected_pTdiff_flag:
                    for item in range(len(Qn_pTdata[:, 0])):
                        self.analyzed_db.insertIntoTable(
                            analyzed_table_pTdiff_name,
                            ((hydroId, urqmdId, pid, weight_type)
                             + tuple(Qn_pTdata[item, :]))
                        )
        self.analyzed_db._dbCon.commit()  # commit changes

    def collect_particle_meanPT(self, particle_name):
        """
            collect particle mean pT without pT cut and rapidity=(-0.5, 0.5)
            in accordance with ALICE identified particle results
        """
         # get pid string
        pid = self.pid_lookup[particle_name]
        pidString = self.getPidString(particle_name)
        analyzed_table_name = 'particle_meanPT'
        # rapidity 
        rap_type = "rapidity"
        rap_lower=-0.5
        rap_upper= 0.5

        # check whether the data are already collected
        collected_flag = True
        if self.analyzed_db.createTableIfNotExists(analyzed_table_name,
                                                   (('pid', 'integer'),
                                                    ('mean_pT_value', 'real'),
                                                    ('mean_pT_error', 'real'))):
            collected_flag = False
        else:
            try_data = array(self.analyzed_db.executeSQLquery(
                "select * from %s where "
                "pid = %d" % (analyzed_table_name, pid)).fetchall())
            if try_data.size == 0: collected_flag = False

        # check whether user wants to update the analyzed data
        if collected_flag:
            print("particle spectra of %s has already been collected!"
                  % particle_name)
            inputval = raw_input(
                "Do you want to delete the existing one and collect again?")
            if inputval.lower() == 'y' or inputval.lower() == 'yes':
                self.analyzed_db.executeSQLquery("delete from %s "
                                                 "where pid = %d" % (
                                                     analyzed_table_name, pid))
                self.analyzed_db._dbCon.commit()  # commit changes
                collected_flag = False

        # collect data loop over all the events
        mean_pT_value = 0.0
        mean_pT_error = 0.0
        if not collected_flag:
            print("collect %s mean pT ..." % particle_name)
            particle_pT = array(self.db.executeSQLquery(
                "select pT from particle_list " 
                "where pid=%d and %g <= %s and %s <= %g"
                %(pid, rap_lower, rap_type, rap_type, rap_upper)).fetchall())
            # calculate mean pt
            if particle_pT.size!=0:
                totalN = len(particle_pT)
                mean_pT_value = sum(particle_pT)/totalN
                mean_pT_error = (sqrt(sum(particle_pT**2.0)/totalN - 
                                      mean_pT_value**2.0)
                                /sqrt(self.totNev -1))
            # insert into table
            self.analyzed_db.insertIntoTable(analyzed_table_name, 
                                            (pid, mean_pT_value, mean_pT_error))
        self.analyzed_db._dbCon.commit()  # commit changes

    ###########################################################################
    # functions to collect two particle correlation
    ########################################################################### 

    def generateAnalyzedDatabase(self):
        self.particle_decay("sigma_0", "lambda", "gamma")
        self.collect_particle_spectra("charged", rap_type='rapidity')
        self.collect_particle_spectra("charged", rap_type='pseudorapidity')
        self.collect_particle_yield_vs_rap("charged",
                                           rap_type='rapidity')
        self.collect_particle_yield_vs_rap("charged",
                                           rap_type='pseudorapidity')

        self.collect_basic_particle_spectra()
        self.collect_flow_Qn_vectors('charged')
        for aPart in ['pion_p', 'kaon_p', 'proton']:
            self.collect_flow_Qn_vectors(aPart)
            self.collect_particle_meanPT(aPart)

        for aPart in ['lambda', 'xi_m', 'omega']:
            self.collect_particle_spectra(aPart, rap_type='rapidity')
        self.collect_flow_Qn_vectors_for_mergedHaron()

    def mergeAnalyzedDatabases(self, toDB, fromDB):
        """
            Merge the analyzed particle database "fromDB" to "toDB"; 
            both are assumed to be databases created from ebe hybrid 
            calculations, which contains exact tables as in 
            analyzed_particles.db.
        """
        for aTable in fromDB.getAllTableNames():
            # first copy table structure
            firstCreation = toDB.createTableIfNotExists(
                aTable, fromDB.getTableInfo(aTable))
            if firstCreation:
                if aTable == 'number_of_events': continue
                # just copy
                toDB.insertIntoTable(aTable, fromDB.selectFromTable(aTable))
            else:  # treatment depends on table type
                # if it's a pid info table, nothing to be done
                if "pid" in aTable: continue
                if aTable == 'number_of_events': continue
                # not a pid info table: shift up hydroEvent_id by the 
                # current existing max
                currentEventIdMax = toDB.selectFromTable(
                    aTable, "max(hydroEvent_id)")[0][0]

                def shiftEID(row):
                    newRow = list(row)
                    newRow[0] += currentEventIdMax
                    return newRow

                toDB.insertIntoTable(
                    aTable, list(map(shiftEID, fromDB.selectFromTable(aTable)))
                )
        toDB.closeConnection()  # commit


def printHelpMessageandQuit():
    print "Usage : "
    print "ParticleReader.py databaseName"
    exit(0)


if __name__ == "__main__":
    if len(argv) < 2:
        printHelpMessageandQuit()
    test = ParticleReader(str(argv[1]))
    test.generateAnalyzedDatabase()
    #for aPart in ['pion_p', 'kaon_p', 'proton']:
    #test.collect_particle_yield_vs_spatial_variable(aPart, 'tau',
    #    linspace(0.0, 15.0, 76), 'rapidity', (-0.5, 0.5))
    #test.collect_particle_yield_vs_spatial_variable(aPart, 'x',
    #    linspace(-13.0, 13.0, 131), 'rapidity', (-0.5, 0.5))
    #test.collect_particle_yield_vs_spatial_variable(aPart, 'eta',
    #    linspace(-2.0, 2.0, 41), 'rapidity', (-0.5, 0.5))
    #test.collect_particle_yield_vs_spatial_variable(aPart, 'tau',
    #    linspace(0.0, 15.0, 76), 'pseudorapidity', (-0.5, 0.5))
    #test.collect_particle_yield_vs_spatial_variable(aPart, 'x',
    #    linspace(-13.0, 13.0, 131), 'pseudorapidity', (-0.5, 0.5))
    #test.collect_particle_yield_vs_spatial_variable(aPart, 'eta',
    #    linspace(-2.0, 2.0, 41), 'pseudorapidity', (-0.5, 0.5))
