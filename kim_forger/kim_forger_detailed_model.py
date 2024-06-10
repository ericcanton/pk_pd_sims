import numpy as np
from scipy.integrate import solve_ivp

initial_conditions = [0.787955, 0.210048, 0.558565, 0.416217, 0.804404, 0.193886, 0.39359, 0.599965, 0.404247, 0.589308,
		4.18947, 0.579108, 8.34963, 1.41168, 12.3735, 4.47054, 31.869, 161.144, 3.76085, 1.09547,
		35.1842, 8.56043, 0.520367, 0.25596, 23.7741, 48.0977, 6.5815, 123.95, 467.844, 0.193289,
		58.2391, 0.0397137, 19.0662, 0.11012, 67.6175, 1.13837, 0.342061, 8.17977, 50.5794, 3.08486,
		0.0985802, 1.30846, 41.1658, 0.0156084, 8.72089, 46.5721, 0.0031953, 1.36799, 0.422898, 1.09254,
		0.00181722, 5.19698e-05, 0.00512223, 0.106575, 0.000833073, 0.0566636, 0.00329174, 1.4996e-05, 0.00543134, 0.0876909,
		0.000333744, 0.0548927, 0.00348546, 1.54312e-05, 0.00510457, 0.0938758, 0.000353611, 0.0535918, 0.914365, 0.664039,
		0.0226162, 0.0241531, 0.00356643, 1.94469e-05, 0.000959363, 0.258442, 0.000985386, 0.0249991, 0.000157103, 1.1242e-05,
		0.00132337, 0.0105914, 0.00023062, 0.025311, 0.00390694, 1.34882e-05, 0.000809309, 0.124712, 0.000416813, 0.0148919,
		0.000188833, 1.79979e-06, 0.00109293, 0.00584769, 4.35837e-05, 0.0166107, 0.00428454, 1.47716e-05, 0.00086132, 0.137696,
		0.000459811, 0.0160349, 0.0002068, 1.94018e-06, 0.0011516, 0.00644626, 4.76349e-05, 0.0177021, 0.00569806, 0.000190832,
		0.0566834, 0.0287569, 0.000413832, 0.0494651, 0.00414505, 0.000386223, 0.124464, 0.00740796, 0.000654378, 0.104089,
		0.00666974, 2.54221e-05, 0.00484524, 0.0828541, 0.000294564, 0.0323573, 0.0012142, 1.1508e-05, 0.00976858, 0.0087474,
		7.98494e-05, 0.0607878, 0.00706385, 2.64292e-05, 0.00442251, 0.0852703, 0.000300906, 0.0301143, 0.00132641, 1.12497e-05,
		0.00878991, 0.0091505, 7.78605e-05, 0.055586, 0.000792825, 2.57804e-05, 0.00334582, 0.0274125, 0.000385498, 0.0589937,
		0.000125512, 4.99769e-05, 0.00735186, 0.00550233, 0.000631987, 0.12768, 0.00252101, 1.00779e-05, 0.00248103, 0.0581233,
		0.000207429, 0.0239137, 0.000305307, 4.90217e-06, 0.00521949, 0.00753943, 6.39585e-05, 0.0471435, 0.00259772, 1.01709e-05,
		0.00224807, 0.0602479, 0.000213449, 0.0225783, 0.000320067, 4.60372e-06, 0.004701, 0.00797183, 6.35261e-05, 0.0440794]

class ForgerKimModel:
    def __init__(self, parameter_values=None):
        """
        Initializes the DetailedModel class.

        Args:
            parameter_values: (Optional) A dictionary of parameter values.
                              If not provided, default values are used.
        """

        # Default parameter values (same as in the MATLAB code)
        self.default_parameters = {
            'trPo': 25.9201, 'trPt': 44.854, 'trRo': 23.0747, 'trRt': 39.9409, 'trB': 46.1038, 'trRev': 102.923, 'trNp': 0.329749, 'tlp': 1.81031, 'tlr': 5.03882, 'tlb': 0.530436,
            'tlrev': 8.90744, 'tlc': 4.64589, 'tlnp': 1.25099, 'agp': 1.3962, 'dg': 2.93521, 'ac': 0.0456572, 'dc': 0.108072, 'ar': 0.0235285, 'dr': 0.605268, 'cbin': 0.0454894,
            'uncbin': 7.27215, 'bbin': 6.92686, 'unbbin': 0.130196, 'cbbin': 6.59924, 'uncbbin': 0.304176, 'ag': 0.162392, 'bin': 6.97166, 'unbin': 0.255032, 'binrev': 0.0120525,
            'unbinrev': 10.9741, 'binr': 6.15445, 'unbinr': 2.91009, 'binc': 0.280863, 'unbinc': 0.00886752, 'binrevb': 0.00626588, 'unbinrevb': 5.30559, 'tmc': 0.16426, 'tmcrev': 9.2631,
            'nl': 0.643086, 'ne': 0.0269078, 'nlrev': 9.63702, 'nerev': 0.0152514, 'lne': 0.594609, 'nlbc': 5.26501, 'hoo': 0.527453, 'hto': 2.45584, 'phos': 0.291429, 'lono': 0.205813,
            'lont': 0.396392, 'lta': 0.607387, 'ltb': 0.013, 'trgto': 0.644602, 'ugto': 0.0625777, 'Nf': 3.35063, 'up': 3.537, 'uro': 0.17491, 'urt': 0.481895, 'umNp': 0.369493,
            'umPo': 0.766962, 'umPt': 0.58892, 'umRo': 0.403425, 'umRt': 0.455544, 'ub': 0.0188002, 'uc': 0.0251651, 'ubc': 0.348829, 'upu': 0.0700322, 'urev': 1.64876, 'uprev': 0.517303,
            'umB': 0.795402, 'umRev': 1.51019
        }

        # If parameter values are provided, update the default values
        if parameter_values:
            self.default_parameters.update(parameter_values)
    
    def simulate(self, t_end=10, num_points=1000):
        """
        Simulates the model.

        Args:
            t_end: The end time of the simulation.
            num_points: The number of time points to simulate.

        Returns:
            A tuple containing the time points and the state variables.
        """

        # Time points
        t = np.linspace(0, t_end, num_points)

        # Initial conditions
        y0 = initial_conditions

        # Solve the ODE system
        solution = solve_ivp(self.rhs, (0, t_end), y0, t_eval=t)

        return solution.t, solution.y

    def rhs(self, t, y):
        """
        Computes the right-hand side (time derivatives) of the ODE system.

        Args:
            t: The current time point.
            y: A numpy array containing the current state variables.

        Returns:
            A numpy array containing the time derivatives of the state variables.
        """

        # Unpack state variables and parameters
        GR, G, GrR, Gr, GcR, Gc, GBR, GB, GBRb, GBb, MnPo, McPo, MnPt, McPt, MnRt, McRt, MnRev, McRev, MnRo, McRo, MnB, McB, MnNp, McNp, B, Cl, BC, cyrev, revn, cyrevg, revng, cyrevgp, revngp, cyrevp, revnp, gto, x00001, x00011, x00100, x00110, x00200, x00210, x01000, x01010, x01011, x02000, x02010, x02011, x10000, x10100, x20000, x20010, x20011, x20100, x20110, x20111, x21000, x21010, x21011, x21100, x21110, x21111, x22000, x22010, x22011, x22100, x22110, x22111, x30000, x30100, x30200, x30300, x40000, x40010, x40011, x40100, x40110, x40111, x40200, x40210, x40211, x40300, x40310, x40311, x41000, x41010, x41011, x41100, x41110, x41111, x41200, x41210, x41211, x41300, x41310, x41311, x42000, x42010, x42011, x42100, x42110, x42111, x42200, x42210, x42211, x42300, x42310, x42311, x50000, x50010, x50011, x50100, x50110, x50111, x50200, x50210, x50211, x50300, x50310, x50311, x51000, x51010, x51011, x51100, x51110, x51111, x51200, x51210, x51211, x51300, x51310, x51311, x52000, x52010, x52011, x52100, x52110, x52111, x52200, x52210, x52211, x52300, x52310, x52311, x60000, x60010, x60011, x60100, x60110, x60111, x60200, x60210, x60211, x60300, x60310, x60311, x61000, x61010, x61011, x61100, x61110, x61111, x61200, x61210, x61211, x61300, x61310, x61311, x62000, x62010, x62011, x62100, x62110, x62111, x62200, x62210, x62211, x62300, x62310, x62311 = y
        
        bin = self.default_parameters['bin']
        unbin = self.default_parameters['unbin']
        unbinr = self.default_parameters['unbinr']
        binr = self.default_parameters['binr']
        unbinc = self.default_parameters['unbinc']
        binc = self.default_parameters['binc']
        unbinrev = self.default_parameters['unbinrev']
        binrev = self.default_parameters['binrev']
        unbinrevb = self.default_parameters['unbinrevb']
        binrevb = self.default_parameters['binrevb']
        trPo = self.default_parameters['trPo']
        tmc = self.default_parameters['tmc']
        umPo = self.default_parameters['umPo']
        trPt = self.default_parameters['trPt']
        umPt = self.default_parameters['umPt']
        trRt = self.default_parameters['trRt']
        umRt = self.default_parameters['umRt']
        trRev = self.default_parameters['trRev']
        umRev = self.default_parameters['umRev']
        trRo = self.default_parameters['trRo']
        umRo = self.default_parameters['umRo']
        trB = self.default_parameters['trB']
        umB = self.default_parameters['umB']
        trNp = self.default_parameters['trNp']
        umNp = self.default_parameters['umNp']
        ub = self.default_parameters['ub']
        uncbin = self.default_parameters['uncbin']
        cbin = self.default_parameters['cbin']
        tlb = self.default_parameters['tlb']
        tlc = self.default_parameters['tlc']
        tlnp = self.default_parameters['tlnp']
        uc = self.default_parameters['uc']
        nlrev = self.default_parameters['nlrev']
        urev = self.default_parameters['urev']
        dg = self.default_parameters['dg']
        ag = self.default_parameters['ag']
        trgto = self.default_parameters['trgto']
        ugto = self.default_parameters['ugto']
        tlrev = self.default_parameters['tlrev']
        nerev = self.default_parameters['nerev']
        Nf = self.default_parameters['Nf']
        upu = self.default_parameters['upu']
        up = self.default_parameters['up']
        uro = self.default_parameters['uro']
        urt = self.default_parameters['urt']
        umNp = self.default_parameters['umNp']
        umPo = self.default_parameters['umPo']
        umPt = self.default_parameters['umPt']
        uprev = self.default_parameters['uprev']
        nlbc = self.default_parameters['nlbc']
        cbbin = self.default_parameters['cbbin']
        unbbin = self.default_parameters['unbbin']
        bbin = self.default_parameters['bbin']
        lne = self.default_parameters['lne']
        dc = self.default_parameters['dc']
        ac = self.default_parameters['ac']
        ar = self.default_parameters['ar']
        agp = self.default_parameters['agp']
        phos = self.default_parameters['phos']
        ubc = self.default_parameters['ubc']
        uncbbin = self.default_parameters['uncbbin']
        tmcrev = self.default_parameters['tmcrev']
        dr = self.default_parameters['dr']
        tlr = self.default_parameters['tlr']
        hoo = self.default_parameters['hoo']
        tlp = self.default_parameters['tlp']
        nl = self.default_parameters['nl']
        ne = self.default_parameters['ne']
        hto = self.default_parameters['hto']



        GR_dot = -(unbin*GR)+bin*(1-G-GR)*(x01011+x02011)
        G_dot = -(unbin*G)+bin*(1-G-GR)*x00011
        GrR_dot = -(unbinr*GrR)+binr*(1-Gr-GrR)*(x01011+x02011)
        Gr_dot = -(unbinr*Gr)+binr*(1-Gr-GrR)*x00011
        GcR_dot = -(unbinc*GcR)+binc*(1-Gc-GcR)*(x01011+x02011)
        Gc_dot = -(unbinc*Gc)+binc*(1-Gc-GcR)*x00011
        GBR_dot = -(unbinrev*GBR)+binrev*GB*(revn+revng+revngp+revnp)
        GB_dot = unbinrev*GBR-binrev*GB*(revn+revng+revngp+revnp)
        GBRb_dot = -(unbinrevb*GBRb)+binrevb*GBb*(revn+revng+revngp+revnp)
        GBb_dot = unbinrevb*GBRb-binrevb*GBb*(revn+revng+revngp+revnp)
        MnPo_dot = trPo*G-tmc*MnPo-umPo*MnPo
        McPo_dot = -(umPo*McPo)+tmc*MnPo
        MnPt_dot = trPt*G-tmc*MnPt-umPt*MnPt
        McPt_dot = -(umPt*McPt)+tmc*MnPt
        MnRt_dot = trRt*Gc-tmc*MnRt-umRt*MnRt
        McRt_dot = -(umRt*McRt)+tmc*MnRt
        MnRev_dot = -(tmcrev*MnRev)-umRev*MnRev+trRev*Gr*x00011
        McRev_dot = -(umRev*McRev)+tmcrev*MnRev
        MnRo_dot = trRo*G*GB-tmc*MnRo-umRo*MnRo
        McRo_dot = -(umRo*McRo)+tmc*MnRo
        MnB_dot = trB*GBb-tmc*MnB-umB*MnB
        McB_dot = -(umB*McB)+tmc*MnB
        MnNp_dot = trNp*GB-tmc*MnNp-umNp*MnNp
        McNp_dot = -(umNp*McNp)+tmc*MnNp
        B_dot = -(ub*B)+uncbin*BC-cbin*B*Cl+tlb*McB
        Cl_dot = tlc+uncbin*BC-uc*Cl-cbin*B*Cl+tlnp*McNp
        BC_dot = -(phos*BC)-ubc*BC-uncbin*BC+cbin*B*Cl
        cyrev_dot = -((nlrev+urev)*cyrev)+dg*cyrevg+tlrev*McRev+nerev*revn-ag*cyrev*x00200
        revn_dot = nlrev*cyrev+(-nerev-urev)*revn+dg*revng-ag*Nf*revn*x00210
        cyrevg_dot = -(cyrevg*(dg+nlrev+urev+gto))+nerev*revng+ag*cyrev*x00200
        revng_dot = nlrev*cyrevg-(dg+nerev+urev+gto)*revng+ag*Nf*revn*x00210
        cyrevgp_dot = -((dg+nlrev+uprev)*cyrevgp)+cyrevg*gto+nerev*revngp
        revngp_dot = nlrev*cyrevgp+gto*revng-(dg+nerev+uprev)*revngp
        cyrevp_dot = dg*cyrevgp-(nlrev+uprev)*cyrevp+nerev*revnp
        revnp_dot = nlrev*cyrevp+dg*revngp-(nerev+uprev)*revnp
        gto_dot = trgto*G*GB-ugto*gto
        x00001_dot = phos*BC-nlbc*x00001-ubc*x00001
        x00011_dot = nlbc*x00001-ubc*x00011+uro*x01011-cbbin*Nf*x00011*(x01010+x02010)+urt*x02011+uncbbin*(x01011+x02011)+upu*(x50011+x50111+x50211+x50311)+up*(x20011+x20111+x40011+x40111+x40211+x40311+x60011+x60111+x60211+x60311)-bbin*Nf*x00011*(x20010+x20110+x21010+x21110+x22010+x22110+x40010+x40110+x40210+x40310+x41010+x41110+x41210+x41310+x42010+x42110+x42210+x42310+x50010+x50110+x50210+x50310+x51010+x51110+x51210+x51310+x52010+x52110+x52210+x52310+x60010+x60110+x60210+x60310+x61010+x61110+x61210+x61310+x62010+x62110+x62210+x62310)+unbbin*(x20011+x20111+x21011+x21111+x22011+x22111+x40011+x40111+x40211+x40311+x41011+x41111+x41211+x41311+x42011+x42111+x42211+x42311+x50011+x50111+x50211+x50311+x51011+x51111+x51211+x51311+x52011+x52111+x52211+x52311+x60011+x60111+x60211+x60311+x61011+x61111+x61211+x61311+x62011+x62111+x62211+x62311)
        x00100_dot = lne*x00110+upu*(x10100+x30100+x30300+x50100+x50300)+up*(x20100+x40100+x40300+x60100+x60300)-ac*x00100*(x10000+x20000+x21000+x22000+x30000+x40000+x41000+x42000+x50000+x51000+x52000+x60000+x61000+x62000)+dc*(x10100+x20100+x21100+x22100+x30100+x40100+x41100+x42100+x50100+x51100+x52100+x60100+x61100+x62100)-ac*x00100*(x30200+x40200+x41200+x42200+x50200+x51200+x52200+x60200+x61200+x62200)+dc*(x30300+x40300+x41300+x42300+x50300+x51300+x52300+x60300+x61300+x62300)
        x00110_dot = -(lne*x00110)+upu*(x50110+x50111+x50310+x50311)+up*(x20110+x20111+x40110+x40111+x40310+x40311+x60110+x60111+x60310+x60311)-ac*Nf*x00110*(x20010+x21010+x22010+x40010+x41010+x42010+x50010+x51010+x52010+x60010+x61010+x62010)-ac*Nf*x00110*(x20011+x21011+x22011+x40011+x41011+x42011+x50011+x51011+x52011+x60011+x61011+x62011)+dc*(x20110+x21110+x22110+x40110+x41110+x42110+x50110+x51110+x52110+x60110+x61110+x62110)+dc*(x20111+x21111+x22111+x40111+x41111+x42111+x50111+x51111+x52111+x60111+x61111+x62111)-ac*Nf*x00110*(x40210+x41210+x42210+x50210+x51210+x52210+x60210+x61210+x62210)-ac*Nf*x00110*(x40211+x41211+x42211+x50211+x51211+x52211+x60211+x61211+x62211)+dc*(x40310+x41310+x42310+x50310+x51310+x52310+x60310+x61310+x62310)+dc*(x40311+x41311+x42311+x50311+x51311+x52311+x60311+x61311+x62311)
        x00200_dot = dg*cyrevg+urev*cyrevg+dg*cyrevgp+uprev*cyrevgp-ag*cyrev*x00200+lne*x00210+upu*(x30200+x30300+x50200+x50300)+up*(x40200+x40300+x60200+x60300)-agp*x00200*(x30000+x30100+x40000+x40100+x41000+x41100+x42000+x42100+x50000+x50100+x51000+x51100+x52000+x52100+x60000+x60100+x61000+x61100+x62000+x62100)+dg*(x30200+x30300+x40200+x40300+x41200+x41300+x42200+x42300+x50200+x50300+x51200+x51300+x52200+x52300+x60200+x60300+x61200+x61300+x62200+x62300)
        x00210_dot = dg*revng+urev*revng+dg*revngp+uprev*revngp-lne*x00210-ag*Nf*revn*x00210+upu*(x50210+x50211+x50310+x50311)+up*(x40210+x40211+x40310+x40311+x60210+x60211+x60310+x60311)-agp*Nf*x00210*(x40010+x40011+x40110+x40111+x41010+x41011+x41110+x41111+x42010+x42011+x42110+x42111+x50010+x50011+x50110+x50111+x51010+x51011+x51110+x51111+x52010+x52011+x52110+x52111+x60010+x60011+x60110+x60111+x61010+x61011+x61110+x61111+x62010+x62011+x62110+x62111)+dg*(x40210+x40211+x40310+x40311+x41210+x41211+x41310+x41311+x42210+x42211+x42310+x42311+x50210+x50211+x50310+x50311+x51210+x51211+x51310+x51311+x52210+x52211+x52310+x52311+x60210+x60211+x60310+x60311+x61210+x61211+x61310+x61311+x62210+x62211+x62310+x62311)
        x01000_dot = tlr*McRo-uro*x01000-ar*x01000*(x20000+x20100+x40000+x40100+x40200+x40300+x50000+x50100+x50200+x50300+x60000+x60100+x60200+x60300)+dr*(x21000+x21100+x41000+x41100+x41200+x41300+x51000+x51100+x51200+x51300+x61000+x61100+x61200+x61300)
        x01010_dot = -(uro*x01010)-cbbin*Nf*x00011*x01010+uncbbin*x01011-ar*Nf*x01010*(x20010+x20110+x40010+x40110+x40210+x40310+x50010+x50110+x50210+x50310+x60010+x60110+x60210+x60310)-ar*Nf*x01010*(x20011+x20111+x40011+x40111+x40211+x40311+x50011+x50111+x50211+x50311+x60011+x60111+x60211+x60311)+dr*(x21010+x21110+x41010+x41110+x41210+x41310+x51010+x51110+x51210+x51310+x61010+x61110+x61210+x61310)+dr*(x21011+x21111+x41011+x41111+x41211+x41311+x51011+x51111+x51211+x51311+x61011+x61111+x61211+x61311)
        x01011_dot = cbbin*Nf*x00011*x01010-uncbbin*x01011-uro*x01011-ar*Nf*x01011*(x20010+x20110+x40010+x40110+x40210+x40310+x50010+x50110+x50210+x50310+x60010+x60110+x60210+x60310)+dr*(x21011+x21111+x41011+x41111+x41211+x41311+x51011+x51111+x51211+x51311+x61011+x61111+x61211+x61311)
        x02000_dot = tlr*McRt-urt*x02000-ar*x02000*(x20000+x20100+x40000+x40100+x40200+x40300+x50000+x50100+x50200+x50300+x60000+x60100+x60200+x60300)+dr*(x22000+x22100+x42000+x42100+x42200+x42300+x52000+x52100+x52200+x52300+x62000+x62100+x62200+x62300)
        x02010_dot = -(urt*x02010)-cbbin*Nf*x00011*x02010+uncbbin*x02011-ar*Nf*x02010*(x20010+x20110+x40010+x40110+x40210+x40310+x50010+x50110+x50210+x50310+x60010+x60110+x60210+x60310)-ar*Nf*x02010*(x20011+x20111+x40011+x40111+x40211+x40311+x50011+x50111+x50211+x50311+x60011+x60111+x60211+x60311)+dr*(x22010+x22110+x42010+x42110+x42210+x42310+x52010+x52110+x52210+x52310+x62010+x62110+x62210+x62310)+dr*(x22011+x22111+x42011+x42111+x42211+x42311+x52011+x52111+x52211+x52311+x62011+x62111+x62211+x62311)
        x02011_dot = cbbin*Nf*x00011*x02010-uncbbin*x02011-urt*x02011-ar*Nf*x02011*(x20010+x20110+x40010+x40110+x40210+x40310+x50010+x50110+x50210+x50310+x60010+x60110+x60210+x60310)+dr*(x22011+x22111+x42011+x42111+x42211+x42311+x52011+x52111+x52211+x52311+x62011+x62111+x62211+x62311)
        x10000_dot = tlp*McPo-upu*x10000-ac*x00100*x10000+dc*x10100
        x10100_dot = ac*x00100*x10000-dc*x10100-hoo*x10100-upu*x10100
        x20000_dot = -(nl*x20000)-up*x20000-ac*x00100*x20000-ar*(x01000+x02000)*x20000+ne*x20010+dc*x20100+dr*(x21000+x22000)
        x20010_dot = nl*x20000-ne*x20010-up*x20010-bbin*Nf*x00011*x20010-ac*Nf*x00110*x20010-ar*Nf*(x01010+x02010)*x20010-ar*Nf*(x01011+x02011)*x20010+ubc*x20011+unbbin*x20011+dc*x20110+dr*(x21010+x22010)+dr*(x21011+x22011)
        x20011_dot = bbin*Nf*x00011*x20010-ubc*x20011-unbbin*x20011-up*x20011-ac*Nf*x00110*x20011-ar*Nf*(x01010+x02010)*x20011+dc*x20111+dr*(x21011+x22011)
        x20100_dot = hoo*x10100+ac*x00100*x20000-dc*x20100-nl*x20100-up*x20100-ar*(x01000+x02000)*x20100+ne*x20110+dr*(x21100+x22100)
        x20110_dot = ac*Nf*x00110*x20010+nl*x20100-dc*x20110-ne*x20110-up*x20110-bbin*Nf*x00011*x20110-ar*Nf*(x01010+x02010)*x20110-ar*Nf*(x01011+x02011)*x20110+ubc*x20111+unbbin*x20111+dr*(x21110+x22110)+dr*(x21111+x22111)
        x20111_dot = ac*Nf*x00110*x20011+bbin*Nf*x00011*x20110-dc*x20111-ubc*x20111-unbbin*x20111-up*x20111-ar*Nf*(x01010+x02010)*x20111+dr*(x21111+x22111)
        x21000_dot = ar*x01000*x20000-dr*x21000-nl*x21000-ac*x00100*x21000+ne*x21010+dc*x21100
        x21010_dot = ar*Nf*x01010*x20010+nl*x21000-dr*x21010-ne*x21010-bbin*Nf*x00011*x21010-ac*Nf*x00110*x21010+unbbin*x21011+dc*x21110
        x21011_dot = ar*Nf*x01011*x20010+ar*Nf*x01010*x20011+bbin*Nf*x00011*x21010-2*dr*x21011-unbbin*x21011-ac*Nf*x00110*x21011+dc*x21111
        x21100_dot = ar*x01000*x20100+ac*x00100*x21000-dc*x21100-dr*x21100-nl*x21100+ne*x21110
        x21110_dot = ar*Nf*x01010*x20110+ac*Nf*x00110*x21010+nl*x21100-dc*x21110-dr*x21110-ne*x21110-bbin*Nf*x00011*x21110+unbbin*x21111
        x21111_dot = ar*Nf*x01011*x20110+ar*Nf*x01010*x20111+ac*Nf*x00110*x21011+bbin*Nf*x00011*x21110-dc*x21111-2*dr*x21111-unbbin*x21111
        x22000_dot = ar*x02000*x20000-dr*x22000-nl*x22000-ac*x00100*x22000+ne*x22010+dc*x22100
        x22010_dot = ar*Nf*x02010*x20010+nl*x22000-dr*x22010-ne*x22010-bbin*Nf*x00011*x22010-ac*Nf*x00110*x22010+unbbin*x22011+dc*x22110
        x22011_dot = ar*Nf*x02011*x20010+ar*Nf*x02010*x20011+bbin*Nf*x00011*x22010-2*dr*x22011-unbbin*x22011-ac*Nf*x00110*x22011+dc*x22111
        x22100_dot = ar*x02000*x20100+ac*x00100*x22000-dc*x22100-dr*x22100-nl*x22100+ne*x22110
        x22110_dot = ar*Nf*x02010*x20110+ac*Nf*x00110*x22010+nl*x22100-dc*x22110-dr*x22110-ne*x22110-bbin*Nf*x00011*x22110+unbbin*x22111
        x22111_dot = ar*Nf*x02011*x20110+ar*Nf*x02010*x20111+ac*Nf*x00110*x22011+bbin*Nf*x00011*x22110-dc*x22111-2*dr*x22111-unbbin*x22111
        x30000_dot = tlp*McPt-upu*x30000-ac*x00100*x30000-agp*x00200*x30000+dc*x30100+dg*x30200
        x30100_dot = ac*x00100*x30000-dc*x30100-hto*x30100-upu*x30100-agp*x00200*x30100+dg*x30300
        x30200_dot = agp*x00200*x30000-dg*x30200-upu*x30200-gto*x30200-ac*x00100*x30200+dc*x30300
        x30300_dot = agp*x00200*x30100+ac*x00100*x30200-dc*x30300-dg*x30300-hto*x30300-upu*x30300-gto*x30300
        x40000_dot = -(nl*x40000)-up*x40000-ac*x00100*x40000-agp*x00200*x40000-ar*(x01000+x02000)*x40000+ne*x40010+dc*x40100+dg*x40200+dr*(x41000+x42000)
        x40010_dot = nl*x40000-ne*x40010-up*x40010-bbin*Nf*x00011*x40010-ac*Nf*x00110*x40010-agp*Nf*x00210*x40010-ar*Nf*(x01010+x02010)*x40010-ar*Nf*(x01011+x02011)*x40010+ubc*x40011+unbbin*x40011+dc*x40110+dg*x40210+dr*(x41010+x42010)+dr*(x41011+x42011)
        x40011_dot = bbin*Nf*x00011*x40010-ubc*x40011-unbbin*x40011-up*x40011-ac*Nf*x00110*x40011-agp*Nf*x00210*x40011-ar*Nf*(x01010+x02010)*x40011+dc*x40111+dg*x40211+dr*(x41011+x42011)
        x40100_dot = hto*x30100+ac*x00100*x40000-dc*x40100-nl*x40100-up*x40100-agp*x00200*x40100-ar*(x01000+x02000)*x40100+ne*x40110+dg*x40300+dr*(x41100+x42100)
        x40110_dot = ac*Nf*x00110*x40010+nl*x40100-dc*x40110-ne*x40110-up*x40110-bbin*Nf*x00011*x40110-agp*Nf*x00210*x40110-ar*Nf*(x01010+x02010)*x40110-ar*Nf*(x01011+x02011)*x40110+ubc*x40111+unbbin*x40111+dg*x40310+dr*(x41110+x42110)+dr*(x41111+x42111)
        x40111_dot = ac*Nf*x00110*x40011+bbin*Nf*x00011*x40110-dc*x40111-ubc*x40111-unbbin*x40111-up*x40111-agp*Nf*x00210*x40111-ar*Nf*(x01010+x02010)*x40111+dg*x40311+dr*(x41111+x42111)
        x40200_dot = agp*x00200*x40000-dg*x40200-nl*x40200-up*x40200-gto*x40200-ac*x00100*x40200-ar*(x01000+x02000)*x40200+ne*x40210+dc*x40300+dr*(x41200+x42200)
        x40210_dot = agp*Nf*x00210*x40010+nl*x40200-dg*x40210-ne*x40210-up*x40210-gto*x40210-bbin*Nf*x00011*x40210-ac*Nf*x00110*x40210-ar*Nf*(x01010+x02010)*x40210-ar*Nf*(x01011+x02011)*x40210+ubc*x40211+unbbin*x40211+dc*x40310+dr*(x41210+x42210)+dr*(x41211+x42211)
        x40211_dot = agp*Nf*x00210*x40011+bbin*Nf*x00011*x40210-dg*x40211-ubc*x40211-unbbin*x40211-up*x40211-gto*x40211-ac*Nf*x00110*x40211-ar*Nf*(x01010+x02010)*x40211+dc*x40311+dr*(x41211+x42211)
        x40300_dot = hto*x30300+agp*x00200*x40100+ac*x00100*x40200-dc*x40300-dg*x40300-nl*x40300-up*x40300-gto*x40300-ar*(x01000+x02000)*x40300+ne*x40310+dr*(x41300+x42300)
        x40310_dot = agp*Nf*x00210*x40110+ac*Nf*x00110*x40210+nl*x40300-dc*x40310-dg*x40310-ne*x40310-up*x40310-gto*x40310-bbin*Nf*x00011*x40310-ar*Nf*(x01010+x02010)*x40310-ar*Nf*(x01011+x02011)*x40310+ubc*x40311+unbbin*x40311+dr*(x41310+x42310)+dr*(x41311+x42311)
        x40311_dot = agp*Nf*x00210*x40111+ac*Nf*x00110*x40211+bbin*Nf*x00011*x40310-dc*x40311-dg*x40311-ubc*x40311-unbbin*x40311-up*x40311-gto*x40311-ar*Nf*(x01010+x02010)*x40311+dr*(x41311+x42311)
        x41000_dot = ar*x01000*x40000-dr*x41000-nl*x41000-ac*x00100*x41000-agp*x00200*x41000+ne*x41010+dc*x41100+dg*x41200
        x41010_dot = ar*Nf*x01010*x40010+nl*x41000-dr*x41010-ne*x41010-bbin*Nf*x00011*x41010-ac*Nf*x00110*x41010-agp*Nf*x00210*x41010+unbbin*x41011+dc*x41110+dg*x41210
        x41011_dot = ar*Nf*x01011*x40010+ar*Nf*x01010*x40011+bbin*Nf*x00011*x41010-2*dr*x41011-unbbin*x41011-ac*Nf*x00110*x41011-agp*Nf*x00210*x41011+dc*x41111+dg*x41211
        x41100_dot = ar*x01000*x40100+ac*x00100*x41000-dc*x41100-dr*x41100-nl*x41100-agp*x00200*x41100+ne*x41110+dg*x41300
        x41110_dot = ar*Nf*x01010*x40110+ac*Nf*x00110*x41010+nl*x41100-dc*x41110-dr*x41110-ne*x41110-bbin*Nf*x00011*x41110-agp*Nf*x00210*x41110+unbbin*x41111+dg*x41310
        x41111_dot = ar*Nf*x01011*x40110+ar*Nf*x01010*x40111+ac*Nf*x00110*x41011+bbin*Nf*x00011*x41110-dc*x41111-2*dr*x41111-unbbin*x41111-agp*Nf*x00210*x41111+dg*x41311
        x41200_dot = ar*x01000*x40200+agp*x00200*x41000-dg*x41200-dr*x41200-nl*x41200-gto*x41200-ac*x00100*x41200+ne*x41210+dc*x41300
        x41210_dot = ar*Nf*x01010*x40210+agp*Nf*x00210*x41010+nl*x41200-dg*x41210-dr*x41210-ne*x41210-gto*x41210-bbin*Nf*x00011*x41210-ac*Nf*x00110*x41210+unbbin*x41211+dc*x41310
        x41211_dot = ar*Nf*x01011*x40210+ar*Nf*x01010*x40211+agp*Nf*x00210*x41011+bbin*Nf*x00011*x41210-dg*x41211-2*dr*x41211-unbbin*x41211-gto*x41211-ac*Nf*x00110*x41211+dc*x41311
        x41300_dot = ar*x01000*x40300+agp*x00200*x41100+ac*x00100*x41200-dc*x41300-dg*x41300-dr*x41300-nl*x41300-gto*x41300+ne*x41310
        x41310_dot = ar*Nf*x01010*x40310+agp*Nf*x00210*x41110+ac*Nf*x00110*x41210+nl*x41300-dc*x41310-dg*x41310-dr*x41310-ne*x41310-gto*x41310-bbin*Nf*x00011*x41310+unbbin*x41311
        x41311_dot = ar*Nf*x01011*x40310+ar*Nf*x01010*x40311+agp*Nf*x00210*x41111+ac*Nf*x00110*x41211+bbin*Nf*x00011*x41310-dc*x41311-dg*x41311-2*dr*x41311-unbbin*x41311-gto*x41311
        x42000_dot = ar*x02000*x40000-dr*x42000-nl*x42000-ac*x00100*x42000-agp*x00200*x42000+ne*x42010+dc*x42100+dg*x42200
        x42010_dot = ar*Nf*x02010*x40010+nl*x42000-dr*x42010-ne*x42010-bbin*Nf*x00011*x42010-ac*Nf*x00110*x42010-agp*Nf*x00210*x42010+unbbin*x42011+dc*x42110+dg*x42210
        x42011_dot = ar*Nf*x02011*x40010+ar*Nf*x02010*x40011+bbin*Nf*x00011*x42010-2*dr*x42011-unbbin*x42011-ac*Nf*x00110*x42011-agp*Nf*x00210*x42011+dc*x42111+dg*x42211
        x42100_dot = ar*x02000*x40100+ac*x00100*x42000-dc*x42100-dr*x42100-nl*x42100-agp*x00200*x42100+ne*x42110+dg*x42300
        x42110_dot = ar*Nf*x02010*x40110+ac*Nf*x00110*x42010+nl*x42100-dc*x42110-dr*x42110-ne*x42110-bbin*Nf*x00011*x42110-agp*Nf*x00210*x42110+unbbin*x42111+dg*x42310
        x42111_dot = ar*Nf*x02011*x40110+ar*Nf*x02010*x40111+ac*Nf*x00110*x42011+bbin*Nf*x00011*x42110-dc*x42111-2*dr*x42111-unbbin*x42111-agp*Nf*x00210*x42111+dg*x42311
        x42200_dot = ar*x02000*x40200+agp*x00200*x42000-dg*x42200-dr*x42200-nl*x42200-gto*x42200-ac*x00100*x42200+ne*x42210+dc*x42300
        x42210_dot = ar*Nf*x02010*x40210+agp*Nf*x00210*x42010+nl*x42200-dg*x42210-dr*x42210-ne*x42210-gto*x42210-bbin*Nf*x00011*x42210-ac*Nf*x00110*x42210+unbbin*x42211+dc*x42310
        x42211_dot = ar*Nf*x02011*x40210+ar*Nf*x02010*x40211+agp*Nf*x00210*x42011+bbin*Nf*x00011*x42210-dg*x42211-2*dr*x42211-unbbin*x42211-gto*x42211-ac*Nf*x00110*x42211+dc*x42311
        x42300_dot = ar*x02000*x40300+agp*x00200*x42100+ac*x00100*x42200-dc*x42300-dg*x42300-dr*x42300-nl*x42300-gto*x42300+ne*x42310
        x42310_dot = ar*Nf*x02010*x40310+agp*Nf*x00210*x42110+ac*Nf*x00110*x42210+nl*x42300-dc*x42310-dg*x42310-dr*x42310-ne*x42310-gto*x42310-bbin*Nf*x00011*x42310+unbbin*x42311
        x42311_dot = ar*Nf*x02011*x40310+ar*Nf*x02010*x40311+agp*Nf*x00210*x42111+ac*Nf*x00110*x42211+bbin*Nf*x00011*x42310-dc*x42311-dg*x42311-2*dr*x42311-unbbin*x42311-gto*x42311
        x50000_dot = -(nl*x50000)-upu*x50000-ac*x00100*x50000-agp*x00200*x50000-ar*(x01000+x02000)*x50000+ne*x50010+dc*x50100+dg*x50200+dr*(x51000+x52000)
        x50010_dot = nl*x50000-ne*x50010-upu*x50010-bbin*Nf*x00011*x50010-ac*Nf*x00110*x50010-agp*Nf*x00210*x50010-ar*Nf*(x01010+x02010)*x50010-ar*Nf*(x01011+x02011)*x50010+ubc*x50011+unbbin*x50011+dc*x50110+dg*x50210+dr*(x51010+x52010)+dr*(x51011+x52011)
        x50011_dot = bbin*Nf*x00011*x50010-ubc*x50011-unbbin*x50011-upu*x50011-ac*Nf*x00110*x50011-agp*Nf*x00210*x50011-ar*Nf*(x01010+x02010)*x50011+dc*x50111+dg*x50211+dr*(x51011+x52011)
        x50100_dot = ac*x00100*x50000-dc*x50100-hto*x50100-nl*x50100-upu*x50100-agp*x00200*x50100-ar*(x01000+x02000)*x50100+ne*x50110+dg*x50300+dr*(x51100+x52100)
        x50110_dot = ac*Nf*x00110*x50010+nl*x50100-dc*x50110-hto*x50110-ne*x50110-upu*x50110-bbin*Nf*x00011*x50110-agp*Nf*x00210*x50110-ar*Nf*(x01010+x02010)*x50110-ar*Nf*(x01011+x02011)*x50110+ubc*x50111+unbbin*x50111+dg*x50310+dr*(x51110+x52110)+dr*(x51111+x52111)
        x50111_dot = ac*Nf*x00110*x50011+bbin*Nf*x00011*x50110-dc*x50111-hto*x50111-ubc*x50111-unbbin*x50111-upu*x50111-agp*Nf*x00210*x50111-ar*Nf*(x01010+x02010)*x50111+dg*x50311+dr*(x51111+x52111)
        x50200_dot = gto*x30200+agp*x00200*x50000-dg*x50200-nl*x50200-upu*x50200-ac*x00100*x50200-ar*(x01000+x02000)*x50200+ne*x50210+dc*x50300+dr*(x51200+x52200)
        x50210_dot = agp*Nf*x00210*x50010+nl*x50200-dg*x50210-ne*x50210-upu*x50210-bbin*Nf*x00011*x50210-ac*Nf*x00110*x50210-ar*Nf*(x01010+x02010)*x50210-ar*Nf*(x01011+x02011)*x50210+ubc*x50211+unbbin*x50211+dc*x50310+dr*(x51210+x52210)+dr*(x51211+x52211)
        x50211_dot = agp*Nf*x00210*x50011+bbin*Nf*x00011*x50210-dg*x50211-ubc*x50211-unbbin*x50211-upu*x50211-ac*Nf*x00110*x50211-ar*Nf*(x01010+x02010)*x50211+dc*x50311+dr*(x51211+x52211)
        x50300_dot = gto*x30300+agp*x00200*x50100+ac*x00100*x50200-dc*x50300-dg*x50300-hto*x50300-nl*x50300-upu*x50300-ar*(x01000+x02000)*x50300+ne*x50310+dr*(x51300+x52300)
        x50310_dot = agp*Nf*x00210*x50110+ac*Nf*x00110*x50210+nl*x50300-dc*x50310-dg*x50310-hto*x50310-ne*x50310-upu*x50310-bbin*Nf*x00011*x50310-ar*Nf*(x01010+x02010)*x50310-ar*Nf*(x01011+x02011)*x50310+ubc*x50311+unbbin*x50311+dr*(x51310+x52310)+dr*(x51311+x52311)
        x50311_dot = agp*Nf*x00210*x50111+ac*Nf*x00110*x50211+bbin*Nf*x00011*x50310-dc*x50311-dg*x50311-hto*x50311-ubc*x50311-unbbin*x50311-upu*x50311-ar*Nf*(x01010+x02010)*x50311+dr*(x51311+x52311)
        x51000_dot = ar*x01000*x50000-dr*x51000-nl*x51000-ac*x00100*x51000-agp*x00200*x51000+ne*x51010+dc*x51100+dg*x51200
        x51010_dot = ar*Nf*x01010*x50010+nl*x51000-dr*x51010-ne*x51010-bbin*Nf*x00011*x51010-ac*Nf*x00110*x51010-agp*Nf*x00210*x51010+unbbin*x51011+dc*x51110+dg*x51210
        x51011_dot = ar*Nf*x01011*x50010+ar*Nf*x01010*x50011+bbin*Nf*x00011*x51010-2*dr*x51011-unbbin*x51011-ac*Nf*x00110*x51011-agp*Nf*x00210*x51011+dc*x51111+dg*x51211
        x51100_dot = ar*x01000*x50100+ac*x00100*x51000-dc*x51100-dr*x51100-nl*x51100-agp*x00200*x51100+ne*x51110+dg*x51300
        x51110_dot = ar*Nf*x01010*x50110+ac*Nf*x00110*x51010+nl*x51100-dc*x51110-dr*x51110-ne*x51110-bbin*Nf*x00011*x51110-agp*Nf*x00210*x51110+unbbin*x51111+dg*x51310
        x51111_dot = ar*Nf*x01011*x50110+ar*Nf*x01010*x50111+ac*Nf*x00110*x51011+bbin*Nf*x00011*x51110-dc*x51111-2*dr*x51111-unbbin*x51111-agp*Nf*x00210*x51111+dg*x51311
        x51200_dot = ar*x01000*x50200+agp*x00200*x51000-dg*x51200-dr*x51200-nl*x51200-ac*x00100*x51200+ne*x51210+dc*x51300
        x51210_dot = ar*Nf*x01010*x50210+agp*Nf*x00210*x51010+nl*x51200-dg*x51210-dr*x51210-ne*x51210-bbin*Nf*x00011*x51210-ac*Nf*x00110*x51210+unbbin*x51211+dc*x51310
        x51211_dot = ar*Nf*x01011*x50210+ar*Nf*x01010*x50211+agp*Nf*x00210*x51011+bbin*Nf*x00011*x51210-dg*x51211-2*dr*x51211-unbbin*x51211-ac*Nf*x00110*x51211+dc*x51311
        x51300_dot = ar*x01000*x50300+agp*x00200*x51100+ac*x00100*x51200-dc*x51300-dg*x51300-dr*x51300-nl*x51300+ne*x51310
        x51310_dot = ar*Nf*x01010*x50310+agp*Nf*x00210*x51110+ac*Nf*x00110*x51210+nl*x51300-dc*x51310-dg*x51310-dr*x51310-ne*x51310-bbin*Nf*x00011*x51310+unbbin*x51311
        x51311_dot = ar*Nf*x01011*x50310+ar*Nf*x01010*x50311+agp*Nf*x00210*x51111+ac*Nf*x00110*x51211+bbin*Nf*x00011*x51310-dc*x51311-dg*x51311-2*dr*x51311-unbbin*x51311
        x52000_dot = ar*x02000*x50000-dr*x52000-nl*x52000-ac*x00100*x52000-agp*x00200*x52000+ne*x52010+dc*x52100+dg*x52200
        x52010_dot = ar*Nf*x02010*x50010+nl*x52000-dr*x52010-ne*x52010-bbin*Nf*x00011*x52010-ac*Nf*x00110*x52010-agp*Nf*x00210*x52010+unbbin*x52011+dc*x52110+dg*x52210
        x52011_dot = ar*Nf*x02011*x50010+ar*Nf*x02010*x50011+bbin*Nf*x00011*x52010-2*dr*x52011-unbbin*x52011-ac*Nf*x00110*x52011-agp*Nf*x00210*x52011+dc*x52111+dg*x52211
        x52100_dot = ar*x02000*x50100+ac*x00100*x52000-dc*x52100-dr*x52100-nl*x52100-agp*x00200*x52100+ne*x52110+dg*x52300
        x52110_dot = ar*Nf*x02010*x50110+ac*Nf*x00110*x52010+nl*x52100-dc*x52110-dr*x52110-ne*x52110-bbin*Nf*x00011*x52110-agp*Nf*x00210*x52110+unbbin*x52111+dg*x52310
        x52111_dot = ar*Nf*x02011*x50110+ar*Nf*x02010*x50111+ac*Nf*x00110*x52011+bbin*Nf*x00011*x52110-dc*x52111-2*dr*x52111-unbbin*x52111-agp*Nf*x00210*x52111+dg*x52311
        x52200_dot = ar*x02000*x50200+agp*x00200*x52000-dg*x52200-dr*x52200-nl*x52200-ac*x00100*x52200+ne*x52210+dc*x52300
        x52210_dot = ar*Nf*x02010*x50210+agp*Nf*x00210*x52010+nl*x52200-dg*x52210-dr*x52210-ne*x52210-bbin*Nf*x00011*x52210-ac*Nf*x00110*x52210+unbbin*x52211+dc*x52310
        x52211_dot = ar*Nf*x02011*x50210+ar*Nf*x02010*x50211+agp*Nf*x00210*x52011+bbin*Nf*x00011*x52210-dg*x52211-2*dr*x52211-unbbin*x52211-ac*Nf*x00110*x52211+dc*x52311
        x52300_dot = ar*x02000*x50300+agp*x00200*x52100+ac*x00100*x52200-dc*x52300-dg*x52300-dr*x52300-nl*x52300+ne*x52310
        x52310_dot = ar*Nf*x02010*x50310+agp*Nf*x00210*x52110+ac*Nf*x00110*x52210+nl*x52300-dc*x52310-dg*x52310-dr*x52310-ne*x52310-bbin*Nf*x00011*x52310+unbbin*x52311
        x52311_dot = ar*Nf*x02011*x50310+ar*Nf*x02010*x50311+agp*Nf*x00210*x52111+ac*Nf*x00110*x52211+bbin*Nf*x00011*x52310-dc*x52311-dg*x52311-2*dr*x52311-unbbin*x52311
        x60000_dot = -(nl*x60000)-up*x60000-ac*x00100*x60000-agp*x00200*x60000-ar*(x01000+x02000)*x60000+ne*x60010+dc*x60100+dg*x60200+dr*(x61000+x62000)
        x60010_dot = nl*x60000-ne*x60010-up*x60010-bbin*Nf*x00011*x60010-ac*Nf*x00110*x60010-agp*Nf*x00210*x60010-ar*Nf*(x01010+x02010)*x60010-ar*Nf*(x01011+x02011)*x60010+ubc*x60011+unbbin*x60011+dc*x60110+dg*x60210+dr*(x61010+x62010)+dr*(x61011+x62011)
        x60011_dot = bbin*Nf*x00011*x60010-ubc*x60011-unbbin*x60011-up*x60011-ac*Nf*x00110*x60011-agp*Nf*x00210*x60011-ar*Nf*(x01010+x02010)*x60011+dc*x60111+dg*x60211+dr*(x61011+x62011)
        x60100_dot = hto*x50100+ac*x00100*x60000-dc*x60100-nl*x60100-up*x60100-agp*x00200*x60100-ar*(x01000+x02000)*x60100+ne*x60110+dg*x60300+dr*(x61100+x62100)
        x60110_dot = hto*x50110+ac*Nf*x00110*x60010+nl*x60100-dc*x60110-ne*x60110-up*x60110-bbin*Nf*x00011*x60110-agp*Nf*x00210*x60110-ar*Nf*(x01010+x02010)*x60110-ar*Nf*(x01011+x02011)*x60110+ubc*x60111+unbbin*x60111+dg*x60310+dr*(x61110+x62110)+dr*(x61111+x62111)
        x60111_dot = hto*x50111+ac*Nf*x00110*x60011+bbin*Nf*x00011*x60110-dc*x60111-ubc*x60111-unbbin*x60111-up*x60111-agp*Nf*x00210*x60111-ar*Nf*(x01010+x02010)*x60111+dg*x60311+dr*(x61111+x62111)
        x60200_dot = gto*x40200+agp*x00200*x60000-dg*x60200-nl*x60200-up*x60200-ac*x00100*x60200-ar*(x01000+x02000)*x60200+ne*x60210+dc*x60300+dr*(x61200+x62200)
        x60210_dot = gto*x40210+agp*Nf*x00210*x60010+nl*x60200-dg*x60210-ne*x60210-up*x60210-bbin*Nf*x00011*x60210-ac*Nf*x00110*x60210-ar*Nf*(x01010+x02010)*x60210-ar*Nf*(x01011+x02011)*x60210+ubc*x60211+unbbin*x60211+dc*x60310+dr*(x61210+x62210)+dr*(x61211+x62211)
        x60211_dot = gto*x40211+agp*Nf*x00210*x60011+bbin*Nf*x00011*x60210-dg*x60211-ubc*x60211-unbbin*x60211-up*x60211-ac*Nf*x00110*x60211-ar*Nf*(x01010+x02010)*x60211+dc*x60311+dr*(x61211+x62211)
        x60300_dot = gto*x40300+hto*x50300+agp*x00200*x60100+ac*x00100*x60200-dc*x60300-dg*x60300-nl*x60300-up*x60300-ar*(x01000+x02000)*x60300+ne*x60310+dr*(x61300+x62300)
        x60310_dot = gto*x40310+hto*x50310+agp*Nf*x00210*x60110+ac*Nf*x00110*x60210+nl*x60300-dc*x60310-dg*x60310-ne*x60310-up*x60310-bbin*Nf*x00011*x60310-ar*Nf*(x01010+x02010)*x60310-ar*Nf*(x01011+x02011)*x60310+ubc*x60311+unbbin*x60311+dr*(x61310+x62310)+dr*(x61311+x62311)
        x60311_dot = gto*x40311+hto*x50311+agp*Nf*x00210*x60111+ac*Nf*x00110*x60211+bbin*Nf*x00011*x60310-dc*x60311-dg*x60311-ubc*x60311-unbbin*x60311-up*x60311-ar*Nf*(x01010+x02010)*x60311+dr*(x61311+x62311)
        x61000_dot = ar*x01000*x60000-dr*x61000-nl*x61000-ac*x00100*x61000-agp*x00200*x61000+ne*x61010+dc*x61100+dg*x61200
        x61010_dot = ar*Nf*x01010*x60010+nl*x61000-dr*x61010-ne*x61010-bbin*Nf*x00011*x61010-ac*Nf*x00110*x61010-agp*Nf*x00210*x61010+unbbin*x61011+dc*x61110+dg*x61210
        x61011_dot = ar*Nf*x01011*x60010+ar*Nf*x01010*x60011+bbin*Nf*x00011*x61010-2*dr*x61011-unbbin*x61011-ac*Nf*x00110*x61011-agp*Nf*x00210*x61011+dc*x61111+dg*x61211
        x61100_dot = ar*x01000*x60100+ac*x00100*x61000-dc*x61100-dr*x61100-nl*x61100-agp*x00200*x61100+ne*x61110+dg*x61300
        x61110_dot = ar*Nf*x01010*x60110+ac*Nf*x00110*x61010+nl*x61100-dc*x61110-dr*x61110-ne*x61110-bbin*Nf*x00011*x61110-agp*Nf*x00210*x61110+unbbin*x61111+dg*x61310
        x61111_dot = ar*Nf*x01011*x60110+ar*Nf*x01010*x60111+ac*Nf*x00110*x61011+bbin*Nf*x00011*x61110-dc*x61111-2*dr*x61111-unbbin*x61111-agp*Nf*x00210*x61111+dg*x61311
        x61200_dot = gto*x41200+ar*x01000*x60200+agp*x00200*x61000-dg*x61200-dr*x61200-nl*x61200-ac*x00100*x61200+ne*x61210+dc*x61300
        x61210_dot = gto*x41210+ar*Nf*x01010*x60210+agp*Nf*x00210*x61010+nl*x61200-dg*x61210-dr*x61210-ne*x61210-bbin*Nf*x00011*x61210-ac*Nf*x00110*x61210+unbbin*x61211+dc*x61310
        x61211_dot = gto*x41211+ar*Nf*x01011*x60210+ar*Nf*x01010*x60211+agp*Nf*x00210*x61011+bbin*Nf*x00011*x61210-dg*x61211-2*dr*x61211-unbbin*x61211-ac*Nf*x00110*x61211+dc*x61311
        x61300_dot = gto*x41300+ar*x01000*x60300+agp*x00200*x61100+ac*x00100*x61200-dc*x61300-dg*x61300-dr*x61300-nl*x61300+ne*x61310
        x61310_dot = gto*x41310+ar*Nf*x01010*x60310+agp*Nf*x00210*x61110+ac*Nf*x00110*x61210+nl*x61300-dc*x61310-dg*x61310-dr*x61310-ne*x61310-bbin*Nf*x00011*x61310+unbbin*x61311
        x61311_dot = gto*x41311+ar*Nf*x01011*x60310+ar*Nf*x01010*x60311+agp*Nf*x00210*x61111+ac*Nf*x00110*x61211+bbin*Nf*x00011*x61310-dc*x61311-dg*x61311-2*dr*x61311-unbbin*x61311
        x62000_dot = ar*x02000*x60000-dr*x62000-nl*x62000-ac*x00100*x62000-agp*x00200*x62000+ne*x62010+dc*x62100+dg*x62200
        x62010_dot = ar*Nf*x02010*x60010+nl*x62000-dr*x62010-ne*x62010-bbin*Nf*x00011*x62010-ac*Nf*x00110*x62010-agp*Nf*x00210*x62010+unbbin*x62011+dc*x62110+dg*x62210
        x62011_dot = ar*Nf*x02011*x60010+ar*Nf*x02010*x60011+bbin*Nf*x00011*x62010-2*dr*x62011-unbbin*x62011-ac*Nf*x00110*x62011-agp*Nf*x00210*x62011+dc*x62111+dg*x62211
        x62100_dot = ar*x02000*x60100+ac*x00100*x62000-dc*x62100-dr*x62100-nl*x62100-agp*x00200*x62100+ne*x62110+dg*x62300
        x62110_dot = ar*Nf*x02010*x60110+ac*Nf*x00110*x62010+nl*x62100-dc*x62110-dr*x62110-ne*x62110-bbin*Nf*x00011*x62110-agp*Nf*x00210*x62110+unbbin*x62111+dg*x62310
        x62111_dot = ar*Nf*x02011*x60110+ar*Nf*x02010*x60111+ac*Nf*x00110*x62011+bbin*Nf*x00011*x62110-dc*x62111-2*dr*x62111-unbbin*x62111-agp*Nf*x00210*x62111+dg*x62311
        x62200_dot = gto*x42200+ar*x02000*x60200+agp*x00200*x62000-dg*x62200-dr*x62200-nl*x62200-ac*x00100*x62200+ne*x62210+dc*x62300
        x62210_dot = gto*x42210+ar*Nf*x02010*x60210+agp*Nf*x00210*x62010+nl*x62200-dg*x62210-dr*x62210-ne*x62210-bbin*Nf*x00011*x62210-ac*Nf*x00110*x62210+unbbin*x62211+dc*x62310
        x62211_dot = gto*x42211+ar*Nf*x02011*x60210+ar*Nf*x02010*x60211+agp*Nf*x00210*x62011+bbin*Nf*x00011*x62210-dg*x62211-2*dr*x62211-unbbin*x62211-ac*Nf*x00110*x62211+dc*x62311
        x62300_dot = gto*x42300+ar*x02000*x60300+agp*x00200*x62100+ac*x00100*x62200-dc*x62300-dg*x62300-dr*x62300-nl*x62300+ne*x62310
        x62310_dot = gto*x42310+ar*Nf*x02010*x60310+agp*Nf*x00210*x62110+ac*Nf*x00110*x62210+nl*x62300-dc*x62310-dg*x62310-dr*x62310-ne*x62310-bbin*Nf*x00011*x62310+unbbin*x62311
        x62311_dot = gto*x42311+ar*Nf*x02011*x60310+ar*Nf*x02010*x60311+agp*Nf*x00210*x62111+ac*Nf*x00110*x62211+bbin*Nf*x00011*x62310-dc*x62311-dg*x62311-2*dr*x62311-unbbin*x62311

        output = np.zeros_like(y)

        output[0] = GR_dot
        output[1] = G_dot
        output[2] = GrR_dot
        output[3] = Gr_dot
        output[4] = GcR_dot
        output[5] = Gc_dot
        output[6] = GBR_dot
        output[7] = GB_dot
        output[8] = GBRb_dot
        output[9] = GBb_dot
        output[10] = MnPo_dot
        output[11] = McPo_dot
        output[12] = MnPt_dot
        output[13] = McPt_dot
        output[14] = MnRt_dot
        output[15] = McRt_dot
        output[16] = MnRev_dot
        output[17] = McRev_dot
        output[18] = MnRo_dot
        output[19] = McRo_dot
        output[20] = MnB_dot
        output[21] = McB_dot
        output[22] = MnNp_dot
        output[23] = McNp_dot
        output[24] = B_dot
        output[25] = Cl_dot
        output[26] = BC_dot
        output[27] = cyrev_dot
        output[28] = revn_dot
        output[29] = cyrevg_dot
        output[30] = revng_dot
        output[31] = cyrevgp_dot
        output[32] = revngp_dot
        output[33] = cyrevp_dot
        output[34] = revnp_dot
        output[35] = gto_dot
        output[36] = x00001_dot
        output[37] = x00011_dot
        output[38] = x00100_dot
        output[39] = x00110_dot
        output[40] = x00200_dot
        output[41] = x00210_dot
        output[42] = x01000_dot
        output[43] = x01010_dot
        output[44] = x01011_dot
        output[45] = x02000_dot
        output[46] = x02010_dot
        output[47] = x02011_dot
        output[48] = x10000_dot
        output[49] = x10100_dot
        output[50] = x20000_dot
        output[51] = x20010_dot
        output[52] = x20011_dot
        output[53] = x20100_dot
        output[54] = x20110_dot
        output[55] = x20111_dot
        output[56] = x21000_dot
        output[57] = x21010_dot
        output[58] = x21011_dot
        output[59] = x21100_dot
        output[60] = x21110_dot
        output[61] = x21111_dot
        output[62] = x22000_dot
        output[63] = x22010_dot
        output[64] = x22011_dot
        output[65] = x22100_dot
        output[66] = x22110_dot
        output[67] = x22111_dot
        output[68] = x30000_dot
        output[69] = x30100_dot
        output[70] = x30200_dot
        output[71] = x30300_dot
        output[72] = x40000_dot
        output[73] = x40010_dot
        output[74] = x40011_dot
        output[75] = x40100_dot
        output[76] = x40110_dot
        output[77] = x40111_dot
        output[78] = x40200_dot
        output[79] = x40210_dot
        output[80] = x40211_dot
        output[81] = x40300_dot
        output[82] = x40310_dot
        output[83] = x40311_dot
        output[84] = x41000_dot
        output[85] = x41010_dot
        output[86] = x41011_dot
        output[87] = x41100_dot
        output[88] = x41110_dot
        output[89] = x41111_dot
        output[90] = x41200_dot
        output[91] = x41210_dot
        output[92] = x41211_dot
        output[93] = x41300_dot
        output[94] = x41310_dot
        output[95] = x41311_dot
        output[96] = x42000_dot
        output[97] = x42010_dot
        output[98] = x42011_dot
        output[99] = x42100_dot
        output[100] = x42110_dot
        output[101] = x42111_dot
        output[102] = x42200_dot
        output[103] = x42210_dot
        output[104] = x42211_dot
        output[105] = x42300_dot
        output[106] = x42310_dot
        output[107] = x42311_dot
        output[108] = x50000_dot
        output[109] = x50010_dot
        output[110] = x50011_dot
        output[111] = x50100_dot
        output[112] = x50110_dot
        output[113] = x50111_dot
        output[114] = x50200_dot
        output[115] = x50210_dot
        output[116] = x50211_dot
        output[117] = x50300_dot
        output[118] = x50310_dot
        output[119] = x50311_dot
        output[120] = x51000_dot
        output[121] = x51010_dot
        output[122] = x51011_dot
        output[123] = x51100_dot
        output[124] = x51110_dot
        output[125] = x51111_dot
        output[126] = x51200_dot
        output[127] = x51210_dot
        output[128] = x51211_dot
        output[129] = x51300_dot
        output[130] = x51310_dot
        output[131] = x51311_dot
        output[132] = x52000_dot
        output[133] = x52010_dot
        output[134] = x52011_dot
        output[135] = x52100_dot
        output[136] = x52110_dot
        output[137] = x52111_dot
        output[138] = x52200_dot
        output[139] = x52210_dot
        output[140] = x52211_dot
        output[141] = x52300_dot
        output[142] = x52310_dot
        output[143] = x52311_dot
        output[144] = x60000_dot
        output[145] = x60010_dot
        output[146] = x60011_dot
        output[147] = x60100_dot
        output[148] = x60110_dot
        output[149] = x60111_dot
        output[150] = x60200_dot
        output[151] = x60210_dot
        output[152] = x60211_dot
        output[153] = x60300_dot
        output[154] = x60310_dot
        output[155] = x60311_dot
        output[156] = x61000_dot
        output[157] = x61010_dot
        output[158] = x61011_dot
        output[159] = x61100_dot
        output[160] = x61110_dot
        output[161] = x61111_dot
        output[162] = x61200_dot
        output[163] = x61210_dot
        output[164] = x61211_dot
        output[165] = x61300_dot
        output[166] = x61310_dot
        output[167] = x61311_dot
        output[168] = x62000_dot
        output[169] = x62010_dot
        output[170] = x62011_dot
        output[171] = x62100_dot
        output[172] = x62110_dot
        output[173] = x62111_dot
        output[174] = x62200_dot
        output[175] = x62210_dot
        output[176] = x62211_dot
        output[177] = x62300_dot
        output[178] = x62310_dot
        output[179] = x62311_dot

        return output

def main(save_plots: bool = False):
    model = ForgerKimModel()
    t, y = model.simulate(t_end=100, num_points=10_000)

    if save_plots:
        import matplotlib.pyplot as plt
        plot_index = 0
        for plot_index in range(y.shape[0]):
            print_index = plot_index + 1
            print("saving plot", print_index, f"of 180 ({print_index/180*100:.2f}%)")
            plt.plot(t, y[plot_index])
            plt.savefig(f'y_{plot_index}.png')
            plt.close()
    
    print("done")
    return t, y


if __name__ == "__main__":
    main(save_plots=True)