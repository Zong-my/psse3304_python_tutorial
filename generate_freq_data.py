# -*- coding: utf-8 -*-
# load psse dll
import copy
import shutil
import pssepath
pssepath.add_pssepath(33)
import psspy, pssplot, dyntools, redirect

# load other modules
import io
import re
import time
import os, sys
import cStringIO
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from func_timeout import func_timeout, FunctionTimedOut

_i = psspy.getdefaultint()
_f = psspy.getdefaultreal()
_s = psspy.getdefaultchar()


def capture_output(func, *args, **kwargs):
    # 创建一个 cStringIO 对象来捕获输出
    old_stdout = sys.stdout
    new_stdout = cStringIO.StringIO()
    try:
        # 重定向 sys.stdout 到 new_stdout
        sys.stdout = new_stdout
        # 调用目标函数
        result = func(*args, **kwargs)
    finally:
        # 恢复原来的 sys.stdout
        sys.stdout = old_stdout
    # 获取捕获的输出
    output = new_stdout.getvalue()
    new_stdout.close()
    return result, output


def get_system_info(sav=None):
    ierr = [1] * 100
    if sav is not None:
        psspy.case(sav)  # load case information (.sav file)

    # 读取系统当前总负荷和总发电量
    err_load, demand = psspy.systot('LOAD')  # Total system load.
    err_gen, supply = psspy.systot('GEN')  # Total system generation
    demand_mw, supply_mw = demand.real, supply.real  # 系统总负荷, 系统总有功出力.
    try:
        assert supply_mw >= demand_mw, "total_generation < total_load !"
    except Exception as e:
        print(e)
    ierr[2], ierr[3] = err_load, err_gen

    # 读取系统当前各负荷分量和各发电量分量
    ierr[4], (genbuses,) = psspy.amachint(-1, 1, 'NUMBER')
    ierr[5], (genid,) = psspy.amachchar(-1, 1, 'ID')
    # ierr[6], (genpq,) = psspy.amachcplx(-1, 1, 'PQGEN')  # 'O_PQGEN', 该api也可提取发电机参数
    ierr[6], (genp,) = psspy.amachreal(-1, 1, 'PGEN')
    ierr[7], (genpmax,) = psspy.amachreal(-1, 1, 'PMAX')
    generators = zip(genbuses, genid, genp, genpmax)

    ierr[8], (loadbuses,) = psspy.aloadint(-1, 1, 'NUMBER')
    ierr[9], (loadid,) = psspy.aloadchar(-1, 1, 'ID')
    ierr[10], (loadt,) = psspy.aloadcplx(-1, 1, 'TOTALACT')  # TOTALACT, TOTALNOM
    loadp = [lt.real for lt in loadt]
    try:
        assert round(demand_mw, 3) == round(sum(loadp), 3), "demand_mw != sum(loadp) !"
    except Exception as e:
        print(e)
    loads = zip(loadbuses, loadid, loadt)

    ierr[11], (tobus,) = psspy.aflowint(-1, 1, 1, 1, 'TONUMBER')
    ierr[12], (frmbus,) = psspy.aflowint(-1, 1, 1, 1, 'FROMNUMBER')
    ierr[13], cktid = psspy.aflowchar(-1, 1, 1, 1, 'ID')
    branches = zip(frmbus, tobus, cktid[0])
    branch_flows = []
    for branch in branches:
        err, pflow = psspy.brnflo(branch[0], branch[1], branch[2])
        branch_flows.append((branch[0], branch[1], branch[2], pflow))

    return loadbuses, demand_mw, generators, genp, loads, branch_flows


def run_steady_sim(sav, df, sp, co, re_machi_id):
    le_, lz_, rr_, hi_ = sp['le'], sp['lz'], sp['rr'], sp['hi']
    ierr = [1] * 100  # check and record for error codes
    ierr[1] = psspy.case(sav)  # load case information (.sav file)

    """ 运行方式设置 """
    # 读取系统当前总负荷和总发电量
    err_load, demand = psspy.systot('LOAD')  # Total system load.
    err_gen, supply = psspy.systot('GEN')  # Total system generation
    demand_mw, supply_mw = demand.real, supply.real  # 系统总负荷, 系统总有功出力.
    try:
        assert supply_mw >= demand_mw, "total_generation < total_load !"
    except Exception as e:
        print(e)
    ierr[2], ierr[3] = err_load, err_gen

    # 读取系统当前各负荷分量和各发电量分量
    ierr[4], (genbuses,) = psspy.amachint(-1, 1, 'NUMBER')
    ierr[5], (genid,) = psspy.amachchar(-1, 1, 'ID')
    # ierr[6], (genpq,) = psspy.amachcplx(-1, 1, 'PQGEN')  # 'O_PQGEN', 该api也可提取发电机参数
    ierr[6], (genp,) = psspy.amachreal(-1, 1, 'PGEN')
    ierr[7], (genpmax,) = psspy.amachreal(-1, 1, 'PMAX')
    generators = zip(genbuses, genid, genp, genpmax)

    ierr[8], (loadbuses,) = psspy.aloadint(-1, 1, 'NUMBER')
    ierr[9], (loadid,) = psspy.aloadchar(-1, 1, 'ID')
    ierr[10], (loadt,) = psspy.aloadcplx(-1, 1, 'TOTALACT')  # TOTALACT, TOTALNOM
    loadp = [lt.real for lt in loadt]
    try:
        assert round(demand_mw, 3) == round(sum(loadp), 3), "demand_mw != sum(loadp) !"
    except Exception as e:
        print(e)
    loads = zip(loadbuses, loadid, loadp)

    # 负荷水平设置(通过设置IPload)
    # 将此轮负荷水平变动按照原始各负荷分量占比分配
    ierr[11], (loadip,) = psspy.aloadcplx(-1, 1, 'TOTALNOM')  # TOTALACT, TOTALNOM
    ipload = [lp.real for lp in loadip]
    load_ip = zip(loadbuses, ipload)
    for l, i in load_ip:
        err = psspy.load_chng_4(l, r"""1""", [_i, _i, _i, _i, _i, _i], [_f, _f, i * le_ / 100, _f, _f, _f])
        assert err == 0, "load change fault !"

    # 旋转备用设置(修改发电机Pmax)
    # 将此轮旋转备用变动按照原始各发电量分量占比分配
    total_rr = demand_mw * rr_  # 系统总的旋转备用
    for b, i, p, pm in generators:
        err = psspy.machine_chng_2(b, r"""1""", [_i, _i, _i, _i, _i, _i],
                                   [_f, _f, _f, _f, p + (p / sum(genp)) * total_rr,
                                    _f, _f, _f, _f, _f, _f, _f, _f, _f, _f, _f, _f])
        assert err == 0, "machine change fault !"

    # 潮流不稳定直接跳过
    ierr[12] = psspy.fdns([0, 0, 0, 1, 1, 0, 99, 0])  # fixed slope decoupled Newton-Raphson power flow calculation
    # assert ierr[12] == 0, "Error in power flow calculation! \n %s" % out_file
    if ierr[12] == 0:
        return None

    # 重跑潮流后线路及其运行数据
    loadbuses_, demand_mw_, generators_, genp_, loads_, branch_flows_ = get_system_info()

    # 负荷模型设置
    ierr[13] = psspy.cong(0)  # Convert Generators
    ierr[14] = psspy.conl(0, 1, 1, [0, 0], [lz_[1], lz_[0], lz_[1], lz_[0]])  # initialize for load conversion
    ierr[15] = psspy.conl(0, 1, 2, [0, 0], [lz_[1], lz_[0], lz_[1], lz_[0]])  # convert loads; 'PI', 'PZ', 'QI', 'QZ'
    ierr[16] = psspy.conl(0, 1, 3, [0, 0], [lz_[1], lz_[0], lz_[1], lz_[0]])  # postprocessing housekeeping
    ierr[17] = psspy.ordr(1)
    ierr[18] = psspy.fact()
    ierr[19] = psspy.tysl(0)
    ierr[20] = psspy.dyre_new([1, 1, 1, 1], df, "", "", "")

    # 惯性时间常数设置(GENROU:30-38, GENCLS:39)
    h_GENROU, nu = [6.05, 3.41, 6.05, 3.41, 5.016, 3.141, 3.141, 5.32], [30, 31, 32, 34, 35, 36, 37, 38]
    h_GENCLS, ns = [500], [39]
    for u in range(len(nu)):
        err = psspy.change_plmod_con(nu[u], r"""1""", r"""GENROU""", 5, h_GENROU[u] * hi_)
        assert err == 0, "GENROU h change fault! Bus num %s" % nu[u]
    for s in range(len(ns)):
        err = psspy.change_plmod_con(ns[s], r"""1""", r"""GENCLS""", 1, h_GENCLS[s] * hi_)
        assert err == 0, "GENCLS h change fault! Bus num %s" % ns[s]

    """ 扰动故障前置 """
    ierr[21] = psspy.delete_all_plot_channels()  # Delete channels stored memory
    ierr[22] = psspy.set_relang(1, re_machi_id, r"""1""")  # 设置功角相对值, use machine ID(30) as the reference machine

    # Pick Channels
    if co == 'All':  # 2,3,6,12,13,14,15,16,25,26
        # ierr[23] = psspy.chsb(0, 1, [-1, -1, -1, 1, 1, 0])  # ANGLE, machine relative rotor angle (degrees).
        ierr[24] = psspy.chsb(0, 1, [-1, -1, -1, 1, 2, 0])  # PELEC, machine electrical power (pu on SBASE).
        ierr[25] = psspy.chsb(0, 1, [-1, -1, -1, 1, 3, 0])  # QELEC, machine reactive power.
        # ierr[26] = psspy.chsb(0, 1, [-1, -1, -1, 1, 4, 0])  # ETERM, machine terminal voltage (pu).
        # ierr[27] = psspy.chsb(0, 1, [-1, -1, -1, 1, 5, 0])  # EFD, generator main field voltage (pu).
        ierr[28] = psspy.chsb(0, 1, [-1, -1, -1, 1, 6, 0])  # PMECH, turbine mechanical power (pu on MBASE).
        # ierr[29] = psspy.chsb(0, 1, [-1, -1, -1, 1, 7, 0])  # SPEED, machine speed deviation from nominal (pu).
        # ierr[30] = psspy.chsb(0, 1, [-1, -1, -1, 1, 8, 0])  # XADIFD, machine field current (pu).
        # ierr[31] = psspy.chsb(0, 1, [-1, -1, -1, 1, 9, 0])  # ECOMP, voltage regulator compensated voltage (pu).
        # ierr[32] = psspy.chsb(0, 1, [-1, -1, -1, 1, 10, 0])  # VOTHSG, stabilizer output signal (pu).
        # ierr[33] = psspy.chsb(0, 1, [-1, -1, -1, 1, 11, 0])  # VREF, voltage regulator voltage setpoint (pu).
        ierr[34] = psspy.chsb(0, 1, [-1, -1, -1, 1, 12, 0])  # BSFREQ, bus pu frequency deviations.
        ierr[35] = psspy.chsb(0, 1, [-1, -1, -1, 1, 13, 0])  # VOLT, bus pu voltages (complex).
        ierr[36] = psspy.chsb(0, 1, [-1, -1, -1, 1, 14, 0])  # voltage and angle
        ierr[37] = psspy.chsb(0, 1, [-1, -1, -1, 1, 15, 0])  # flow (P).
        ierr[38] = psspy.chsb(0, 1, [-1, -1, -1, 1, 16, 0])  # flow (P and Q).
        # ierr[39] = psspy.chsb(0, 1, [-1, -1, -1, 1, 17, 0])  # flow (MVA).
        # ierr[40] = psspy.chsb(0, 1, [-1, -1, -1, 1, 18, 0])  # apparent impedance (R and X).
        # ierr[43] = psspy.chsb(0, 1, [-1, -1, -1, 1, 21, 0])  # ITERM.
        # ierr[44] = psspy.chsb(0, 1, [-1, -1, -1, 1, 22, 0])  # machine apparent impedance
        # ierr[45] = psspy.chsb(0, 1, [-1, -1, -1, 1, 23, 0])  # VUEL, minimum excitation limiter output signal (pu).
        # ierr[46] = psspy.chsb(0, 1, [-1, -1, -1, 1, 24, 0])  # VOEL, maximum excitation limiter output signal (pu).
        ierr[47] = psspy.chsb(0, 1, [-1, -1, -1, 1, 25, 0])  # PLOAD.
        ierr[48] = psspy.chsb(0, 1, [-1, -1, -1, 1, 26, 0])  # QLOAD.
        # ierr[49] = psspy.chsb(0, 1, [-1, -1, -1, 1, 27, 0])  # GREF, turbine governor reference.
        # ierr[50] = psspy.chsb(0, 1, [-1, -1, -1, 1, 28, 0])  # LCREF, turbine load control reference.
        # ierr[51] = psspy.chsb(0, 1, [-1, -1, -1, 1, 29, 0])  # WVLCTY, wind velocity (m/s).
        # ierr[52] = psspy.chsb(0, 1, [-1, -1, -1, 1, 30, 0])  # WTRBSP, wind turbine rotor speed deviation (pu).
        # ierr[53] = psspy.chsb(0, 1, [-1, -1, -1, 1, 31, 0])  # WPITCH, pitch angle (degrees).
        # ierr[54] = psspy.chsb(0, 1, [-1, -1, -1, 1, 32, 0])  # WAEROT, aerodynamic torque (pu on MBASE).
        # ierr[55] = psspy.chsb(0, 1, [-1, -1, -1, 1, 33, 0])  # WROTRV, rotor voltage (pu on MBASE).
        # ierr[56] = psspy.chsb(0, 1, [-1, -1, -1, 1, 34, 0])  # WROTRI, rotor current (pu on MBASE).
        # ierr[57] = psspy.chsb(0, 1, [-1, -1, -1, 1, 35, 0])  # WPCMND, active power command from wind control (pu on MBASE).
        # ierr[58] = psspy.chsb(0, 1, [-1, -1, -1, 1, 36, 0])  # 36 WQCMND, reactive power command from wind control (pu on MBASE).
        # ierr[59] = psspy.chsb(0, 1, [-1, -1, -1, 1, 37, 0])  # WAUXSG, output of wind auxiliary control (pu on MBASE).
    elif co == 'GVEA':
        ierr[60] = psspy.bsys(1, 0, [0.0, 0.0], 0, [], 9, [325, 33601, 33500, 30110, 30120, 31001, 32001, 32300, 32100],
                              0, [], 0, [])
        ierr[61] = psspy.chsb(1, 0, [-1, -1, -1, 1, 2, 0])  # Machine electrical power
        ierr[62] = psspy.chsb(1, 0, [-1, -1, -1, 1, 12, 0])  # Bus Frequency Deviations (pu)
        ierr[63] = psspy.chsb(1, 0, [-1, -1, -1, 1, 13, 0])  # Bus Voltage and angle (pu)
    print(ierr)

    return demand_mw_, generators_, genp_, loads_, branch_flows_


def circuit_short(basepath, out, tst, tet, fbus, tbus, bid, rt, sav, df, sp, co, re_machi_id):
    le_, lz_, rr_, hi_ = sp['le'], sp['lz'], sp['rr'], sp['hi']
    ierr = [1] * 100  # check and record for error codes
    ierr[1] = psspy.case(sav)  # load case information (.sav file)

    """ 运行方式设置 """
    # 读取系统当前总负荷和总发电量
    err_load, demand = psspy.systot('LOAD')  # Total system load.
    err_gen, supply = psspy.systot('GEN')  # Total system generation
    demand_mw, supply_mw = demand.real, supply.real  # 系统总负荷, 系统总有功出力.
    try:
        assert supply_mw >= demand_mw, "total_generation < total_load !"
    except Exception as e:
        print(e)
    ierr[2], ierr[3] = err_load, err_gen

    # 读取系统当前各负荷分量和各发电量分量
    ierr[4], (genbuses,) = psspy.amachint(-1, 1, 'NUMBER')
    ierr[5], (genid,) = psspy.amachchar(-1, 1, 'ID')
    # ierr[6], (genpq,) = psspy.amachcplx(-1, 1, 'PQGEN')  # 'O_PQGEN', 该api也可提取发电机参数
    ierr[6], (genp,) = psspy.amachreal(-1, 1, 'PGEN')
    ierr[7], (genpmax,) = psspy.amachreal(-1, 1, 'PMAX')
    generators = zip(genbuses, genid, genp, genpmax)

    ierr[8], (loadbuses,) = psspy.aloadint(-1, 1, 'NUMBER')
    ierr[9], (loadid,) = psspy.aloadchar(-1, 1, 'ID')
    ierr[10], (loadt,) = psspy.aloadcplx(-1, 1, 'TOTALACT')  # TOTALACT, TOTALNOM
    loadp = [lt.real for lt in loadt]
    try:
        assert round(demand_mw, 3) == round(sum(loadp), 3), "demand_mw != sum(loadp) !"
    except Exception as e:
        print(e)
    loads = zip(loadbuses, loadid, loadt)

    # 负荷水平设置(通过设置IPload)
    # 将此轮负荷水平变动按照原始各负荷分量占比分配
    ierr[11], (loadip,) = psspy.aloadcplx(-1, 1, 'TOTALNOM')  # TOTALACT, TOTALNOM
    ipload = [lp.real for lp in loadip]
    load_ip = zip(loadbuses, ipload)
    for l, i in load_ip:
        err = psspy.load_chng_4(l, r"""1""", [_i, _i, _i, _i, _i, _i], [_f, _f, i * le_ / 100, _f, _f, _f])
        assert err == 0, "load change fault !"

    # 旋转备用设置(修改发电机Pmax)
    # 将此轮旋转备用变动按照原始各发电量分量占比分配
    total_rr = demand_mw * rr_  # 系统总的旋转备用
    for b, i, p, pm in generators:
        err = psspy.machine_chng_2(b, r"""1""", [_i, _i, _i, _i, _i, _i],
                                   [_f, _f, _f, _f, p + (p / sum(genp)) * total_rr,
                                    _f, _f, _f, _f, _f, _f, _f, _f, _f, _f, _f, _f])
        assert err == 0, "machine change fault !"

    # 潮流不稳定直接跳过
    ierr[12] = psspy.fnsl([0, 0, 0, 1, 1, 0, 99, 0])
    # ierr[12] = psspy.fdns([0, 0, 0, 1, 1, 0, 99, 0])  # fixed slope decoupled Newton-Raphson power flow calculation
    # assert ierr[12] == 0, "Error in power flow calculation! \n %s" % out_file
    if ierr[12] != 0:
        print("%s \npower flow is not converged!" % out)
        return

    # 重跑潮流后线路及其运行数据
    loadbuses_, demand_mw_, generators_, genp_, loads_, branch_flows = get_system_info()

    # 负荷模型设置
    ierr[13] = psspy.cong(0)  # Convert Generators
    ierr[14] = psspy.conl(0, 1, 1, [0, 0], [lz_[1], lz_[0], lz_[1], lz_[0]])  # initialize for load conversion
    ierr[15] = psspy.conl(0, 1, 2, [0, 0], [lz_[1], lz_[0], lz_[1], lz_[0]])  # convert loads; 'PI', 'PZ', 'QI', 'QZ'
    ierr[16] = psspy.conl(0, 1, 3, [0, 0], [lz_[1], lz_[0], lz_[1], lz_[0]])  # postprocessing housekeeping
    ierr[17] = psspy.ordr(1)
    ierr[18] = psspy.fact()
    ierr[19] = psspy.tysl(0)
    ierr[20] = psspy.dyre_new([1, 1, 1, 1], df, "", "", "")

    # 惯性时间常数设置(GENROU:30-38, GENCLS:39)
    h_GENROU, nu = [6.05, 3.41, 6.05, 3.41, 5.016, 3.141, 3.141, 5.32], [30, 31, 32, 34, 35, 36, 37, 38]
    h_GENCLS, ns = [500], [39]
    for u in range(len(nu)):
        err = psspy.change_plmod_con(nu[u], r"""1""", r"""GENROU""", 5, h_GENROU[u] * hi_)
        assert err == 0, "GENROU h change fault! Bus num %s" % nu[u]
    for s in range(len(ns)):
        err = psspy.change_plmod_con(ns[s], r"""1""", r"""GENCLS""", 1, h_GENCLS[s] * hi_)
        assert err == 0, "GENCLS h change fault! Bus num %s" % ns[s]

    """ 扰动故障前置 """
    ierr[22] = psspy.set_relang(1, re_machi_id, r"""1""")  # 设置功角相对值, use machine ID(30) as the reference machine

    # Pick Channels
    if co == 'All':  # 2,3,6,12,13,14,15,16,25,26
        # ierr[23] = psspy.chsb(0, 1, [-1, -1, -1, 1, 1, 0])  # ANGLE, machine relative rotor angle (degrees).
        ierr[24] = psspy.chsb(0, 1, [-1, -1, -1, 1, 2, 0])  # PELEC, machine electrical power (pu on SBASE).
        ierr[25] = psspy.chsb(0, 1, [-1, -1, -1, 1, 3, 0])  # QELEC, machine reactive power.
        # ierr[26] = psspy.chsb(0, 1, [-1, -1, -1, 1, 4, 0])  # ETERM, machine terminal voltage (pu).
        # ierr[27] = psspy.chsb(0, 1, [-1, -1, -1, 1, 5, 0])  # EFD, generator main field voltage (pu).
        ierr[28] = psspy.chsb(0, 1, [-1, -1, -1, 1, 6, 0])  # PMECH, turbine mechanical power (pu on MBASE).
        # ierr[29] = psspy.chsb(0, 1, [-1, -1, -1, 1, 7, 0])  # SPEED, machine speed deviation from nominal (pu).
        # ierr[30] = psspy.chsb(0, 1, [-1, -1, -1, 1, 8, 0])  # XADIFD, machine field current (pu).
        # ierr[31] = psspy.chsb(0, 1, [-1, -1, -1, 1, 9, 0])  # ECOMP, voltage regulator compensated voltage (pu).
        # ierr[32] = psspy.chsb(0, 1, [-1, -1, -1, 1, 10, 0])  # VOTHSG, stabilizer output signal (pu).
        # ierr[33] = psspy.chsb(0, 1, [-1, -1, -1, 1, 11, 0])  # VREF, voltage regulator voltage setpoint (pu).
        ierr[34] = psspy.chsb(0, 1, [-1, -1, -1, 1, 12, 0])  # BSFREQ, bus pu frequency deviations.
        ierr[35] = psspy.chsb(0, 1, [-1, -1, -1, 1, 13, 0])  # VOLT, bus pu voltages (complex).
        ierr[36] = psspy.chsb(0, 1, [-1, -1, -1, 1, 14, 0])  # voltage and angle
        ierr[37] = psspy.chsb(0, 1, [-1, -1, -1, 1, 15, 0])  # flow (P).
        ierr[38] = psspy.chsb(0, 1, [-1, -1, -1, 1, 16, 0])  # flow (P and Q).
        # ierr[39] = psspy.chsb(0, 1, [-1, -1, -1, 1, 17, 0])  # flow (MVA).
        # ierr[40] = psspy.chsb(0, 1, [-1, -1, -1, 1, 18, 0])  # apparent impedance (R and X).
        # ierr[43] = psspy.chsb(0, 1, [-1, -1, -1, 1, 21, 0])  # ITERM.
        # ierr[44] = psspy.chsb(0, 1, [-1, -1, -1, 1, 22, 0])  # machine apparent impedance
        # ierr[45] = psspy.chsb(0, 1, [-1, -1, -1, 1, 23, 0])  # VUEL, minimum excitation limiter output signal (pu).
        # ierr[46] = psspy.chsb(0, 1, [-1, -1, -1, 1, 24, 0])  # VOEL, maximum excitation limiter output signal (pu).
        ierr[47] = psspy.chsb(0, 1, [-1, -1, -1, 1, 25, 0])  # PLOAD.
        ierr[48] = psspy.chsb(0, 1, [-1, -1, -1, 1, 26, 0])  # QLOAD.
        # ierr[49] = psspy.chsb(0, 1, [-1, -1, -1, 1, 27, 0])  # GREF, turbine governor reference.
        # ierr[50] = psspy.chsb(0, 1, [-1, -1, -1, 1, 28, 0])  # LCREF, turbine load control reference.
        # ierr[51] = psspy.chsb(0, 1, [-1, -1, -1, 1, 29, 0])  # WVLCTY, wind velocity (m/s).
        # ierr[52] = psspy.chsb(0, 1, [-1, -1, -1, 1, 30, 0])  # WTRBSP, wind turbine rotor speed deviation (pu).
        # ierr[53] = psspy.chsb(0, 1, [-1, -1, -1, 1, 31, 0])  # WPITCH, pitch angle (degrees).
        # ierr[54] = psspy.chsb(0, 1, [-1, -1, -1, 1, 32, 0])  # WAEROT, aerodynamic torque (pu on MBASE).
        # ierr[55] = psspy.chsb(0, 1, [-1, -1, -1, 1, 33, 0])  # WROTRV, rotor voltage (pu on MBASE).
        # ierr[56] = psspy.chsb(0, 1, [-1, -1, -1, 1, 34, 0])  # WROTRI, rotor current (pu on MBASE).
        # ierr[57] = psspy.chsb(0, 1, [-1, -1, -1, 1, 35, 0])  # WPCMND, active power command from wind control (pu on MBASE).
        # ierr[58] = psspy.chsb(0, 1, [-1, -1, -1, 1, 36, 0])  # 36 WQCMND, reactive power command from wind control (pu on MBASE).
        # ierr[59] = psspy.chsb(0, 1, [-1, -1, -1, 1, 37, 0])  # WAUXSG, output of wind auxiliary control (pu on MBASE).
    elif co == 'GVEA':
        ierr[60] = psspy.bsys(1, 0, [0.0, 0.0], 0, [], 9, [325, 33601, 33500, 30110, 30120, 31001, 32001, 32300, 32100],
                              0, [], 0, [])
        ierr[61] = psspy.chsb(1, 0, [-1, -1, -1, 1, 2, 0])  # Machine electrical power
        ierr[62] = psspy.chsb(1, 0, [-1, -1, -1, 1, 12, 0])  # Bus Frequency Deviations (pu)
        ierr[63] = psspy.chsb(1, 0, [-1, -1, -1, 1, 13, 0])  # Bus Voltage and angle (pu)
    print(ierr)
    ierr = [1] * 20
    ierr[1] = psspy.strt(0, os.path.join(basepath, out))  # (psse33) Initialize dynamic simulation
    ierr[2] = psspy.run(0, tst, 0, 1, 1)  # 设置时短路故障刻
    ierr[3] = psspy.dist_bus_fault(frmbus)  # 短路故障
    ierr[4] = psspy.run(0, tst + 0.08, 0, 1, 1)  # 设置切线时刻
    ierr[5] = psspy.dist_branch_trip(frmbus, tobus, cktid)  # 切短路线
    ierr[6] = psspy.dist_clear_fault(1)  # clears fault
    ierr[7] = psspy.run(0, tst + 0.08 + 1.2, 0, 1, 1)  # 设置合闸时间
    ierr[8] = psspy.dist_branch_close(frmbus, tobus, cktid)  # 合闸
    ierr[9] = psspy.run(0, rt, 0, 1, 1)  # 设置仿真结束时间
    ierr[10] = psspy.delete_all_plot_channels()  # Delete plot channels to get ready for next simulation
    # time.sleep(0.1)
    console_output = tail(log_path, 3)
    console_output = ''.join(console_output)
    print(ierr)
    # Check for errors written in output file
    current_error = 0
    if "Network not converged" in console_output:
        print('Network not converged')
        current_error = 1
    # raise SystemExit #this will quit the program, if the program is called within a larger
    # program, like optimization, you will want to stop PSSe from running using this or
    # have this rerun the program or skip this iteration's results
    elif "NaN" in console_output:
        print("NaN, network is no good")
        current_error = 1
    # raise SystemExit #this will quit the program, if the program is called within a larger
    # program, like optimization, you will want to stop PSSe from running using this or
    # have this rerun the program or skip this iteration's results
    if current_error == 0 and "INITIAL CONDITIONS CHECK O.K." in console_output:
        print("No errors and initial conditions were good.")

    if current_error > 0:
        return

    # Gather the data and output to excel
    data = dyntools.CHNF(os.path.join(basepath, out))  # getting data from channel.out file
    d, e, z = data.get_data()  # gathering data from data in dictionary format
    # z contains the time series output data from each channel
    # e contains the label of each channel
    # d contains the header of the output file (case name, etc.)
    sim_re = fetch_results(d, e, z)
    xlsx = os.path.join(basepath, os.path.join(basepath, out)[:-4] + '.xlsx')
    # Save Data to excel file
    with pd.ExcelWriter(xlsx) as writer:
        for k, v in sim_re.items():
            v.to_excel(writer, sheet_name=k)
    return


def load_change(basepath, out, tst, tet, ld, lodbus, lodid, lodcplx, rt, sav, df, sp, co, re_machi_id):
    le_, lz_, rr_, hi_ = sp['le'], sp['lz'], sp['rr'], sp['hi']
    ierr = [1] * 100  # check and record for error codes
    ierr[1] = psspy.case(sav)  # load case information (.sav file)

    """ 运行方式设置 """
    # 读取系统当前总负荷和总发电量
    err_load, demand = psspy.systot('LOAD')  # Total system load.
    err_gen, supply = psspy.systot('GEN')  # Total system generation
    demand_mw, supply_mw = demand.real, supply.real  # 系统总负荷, 系统总有功出力.
    try:
        assert supply_mw >= demand_mw, "total_generation < total_load !"
    except Exception as e:
        print(e)
    ierr[2], ierr[3] = err_load, err_gen

    # 读取系统当前各负荷分量和各发电量分量
    ierr[4], (genbuses,) = psspy.amachint(-1, 1, 'NUMBER')
    ierr[5], (genid,) = psspy.amachchar(-1, 1, 'ID')
    # ierr[6], (genpq,) = psspy.amachcplx(-1, 1, 'PQGEN')  # 'O_PQGEN', 该api也可提取发电机参数
    ierr[6], (genp,) = psspy.amachreal(-1, 1, 'PGEN')
    ierr[7], (genpmax,) = psspy.amachreal(-1, 1, 'PMAX')
    generators = zip(genbuses, genid, genp, genpmax)

    ierr[8], (loadbuses,) = psspy.aloadint(-1, 1, 'NUMBER')
    ierr[9], (loadid,) = psspy.aloadchar(-1, 1, 'ID')
    ierr[10], (loadt,) = psspy.aloadcplx(-1, 1, 'TOTALACT')  # TOTALACT, TOTALNOM
    loadp = [lt.real for lt in loadt]
    try:
        assert round(demand_mw, 3) == round(sum(loadp), 3), "demand_mw != sum(loadp) !"
    except Exception as e:
        print(e)
    loads = zip(loadbuses, loadid, loadt)

    # 负荷水平设置(通过设置IPload)
    # 将此轮负荷水平变动按照原始各负荷分量占比分配
    ierr[11], (loadip,) = psspy.aloadcplx(-1, 1, 'TOTALNOM')  # TOTALACT, TOTALNOM
    ipload = [lp.real for lp in loadip]
    load_ip = zip(loadbuses, ipload)
    for l, i in load_ip:
        err = psspy.load_chng_4(l, r"""1""", [_i, _i, _i, _i, _i, _i], [_f, _f, i * le_ / 100, _f, _f, _f])
        assert err == 0, "load change fault !"

    # 旋转备用设置(修改发电机Pmax)
    # 将此轮旋转备用变动按照原始各发电量分量占比分配
    total_rr = demand_mw * rr_  # 系统总的旋转备用
    for b, i, p, pm in generators:
        err = psspy.machine_chng_2(b, r"""1""", [_i, _i, _i, _i, _i, _i],
                                   [_f, _f, _f, _f, p + (p / sum(genp)) * total_rr,
                                    _f, _f, _f, _f, _f, _f, _f, _f, _f, _f, _f, _f])
        assert err == 0, "machine change fault !"

    # 潮流不稳定直接跳过
    ierr[12] = psspy.fnsl([0, 0, 0, 1, 1, 0, 99, 0])
    # ierr[12] = psspy.fdns([0, 0, 0, 1, 1, 0, 99, 0])  # fixed slope decoupled Newton-Raphson power flow calculation
    # assert ierr[12] == 0, "Error in power flow calculation! \n %s" % out_file
    if ierr[12] != 0:
        print("%s \npower flow is not converged!" % out)
        return

    # 重跑潮流后线路及其运行数据
    loadbuses_, demand_mw_, generators_, genp_, loads_, branch_flows = get_system_info()

    # 负荷模型设置
    ierr[13] = psspy.cong(0)  # Convert Generators
    ierr[14] = psspy.conl(0, 1, 1, [0, 0], [lz_[1], lz_[0], lz_[1], lz_[0]])  # initialize for load conversion
    ierr[15] = psspy.conl(0, 1, 2, [0, 0], [lz_[1], lz_[0], lz_[1], lz_[0]])  # convert loads; 'PI', 'PZ', 'QI', 'QZ'
    ierr[16] = psspy.conl(0, 1, 3, [0, 0], [lz_[1], lz_[0], lz_[1], lz_[0]])  # postprocessing housekeeping
    ierr[17] = psspy.ordr(1)
    ierr[18] = psspy.fact()
    ierr[19] = psspy.tysl(0)
    ierr[20] = psspy.dyre_new([1, 1, 1, 1], df, "", "", "")

    # 惯性时间常数设置(GENROU:30-38, GENCLS:39)
    h_GENROU, nu = [6.05, 3.41, 6.05, 3.41, 5.016, 3.141, 3.141, 5.32], [30, 31, 32, 34, 35, 36, 37, 38]
    h_GENCLS, ns = [500], [39]
    for u in range(len(nu)):
        err = psspy.change_plmod_con(nu[u], r"""1""", r"""GENROU""", 5, h_GENROU[u] * hi_)
        assert err == 0, "GENROU h change fault! Bus num %s" % nu[u]
    for s in range(len(ns)):
        err = psspy.change_plmod_con(ns[s], r"""1""", r"""GENCLS""", 1, h_GENCLS[s] * hi_)
        assert err == 0, "GENCLS h change fault! Bus num %s" % ns[s]

    """ 扰动故障前置 """
    ierr[22] = psspy.set_relang(1, re_machi_id, r"""1""")  # 设置功角相对值, use machine ID(30) as the reference machine

    # Pick Channels
    if co == 'All':  # 2,3,6,12,13,14,15,16,25,26
        # ierr[23] = psspy.chsb(0, 1, [-1, -1, -1, 1, 1, 0])  # ANGLE, machine relative rotor angle (degrees).
        ierr[24] = psspy.chsb(0, 1, [-1, -1, -1, 1, 2, 0])  # PELEC, machine electrical power (pu on SBASE).
        ierr[25] = psspy.chsb(0, 1, [-1, -1, -1, 1, 3, 0])  # QELEC, machine reactive power.
        # ierr[26] = psspy.chsb(0, 1, [-1, -1, -1, 1, 4, 0])  # ETERM, machine terminal voltage (pu).
        # ierr[27] = psspy.chsb(0, 1, [-1, -1, -1, 1, 5, 0])  # EFD, generator main field voltage (pu).
        ierr[28] = psspy.chsb(0, 1, [-1, -1, -1, 1, 6, 0])  # PMECH, turbine mechanical power (pu on MBASE).
        # ierr[29] = psspy.chsb(0, 1, [-1, -1, -1, 1, 7, 0])  # SPEED, machine speed deviation from nominal (pu).
        # ierr[30] = psspy.chsb(0, 1, [-1, -1, -1, 1, 8, 0])  # XADIFD, machine field current (pu).
        # ierr[31] = psspy.chsb(0, 1, [-1, -1, -1, 1, 9, 0])  # ECOMP, voltage regulator compensated voltage (pu).
        # ierr[32] = psspy.chsb(0, 1, [-1, -1, -1, 1, 10, 0])  # VOTHSG, stabilizer output signal (pu).
        # ierr[33] = psspy.chsb(0, 1, [-1, -1, -1, 1, 11, 0])  # VREF, voltage regulator voltage setpoint (pu).
        ierr[34] = psspy.chsb(0, 1, [-1, -1, -1, 1, 12, 0])  # BSFREQ, bus pu frequency deviations.
        ierr[35] = psspy.chsb(0, 1, [-1, -1, -1, 1, 13, 0])  # VOLT, bus pu voltages (complex).
        ierr[36] = psspy.chsb(0, 1, [-1, -1, -1, 1, 14, 0])  # voltage and angle
        ierr[37] = psspy.chsb(0, 1, [-1, -1, -1, 1, 15, 0])  # flow (P).
        ierr[38] = psspy.chsb(0, 1, [-1, -1, -1, 1, 16, 0])  # flow (P and Q).
        # ierr[39] = psspy.chsb(0, 1, [-1, -1, -1, 1, 17, 0])  # flow (MVA).
        # ierr[40] = psspy.chsb(0, 1, [-1, -1, -1, 1, 18, 0])  # apparent impedance (R and X).
        # ierr[43] = psspy.chsb(0, 1, [-1, -1, -1, 1, 21, 0])  # ITERM.
        # ierr[44] = psspy.chsb(0, 1, [-1, -1, -1, 1, 22, 0])  # machine apparent impedance
        # ierr[45] = psspy.chsb(0, 1, [-1, -1, -1, 1, 23, 0])  # VUEL, minimum excitation limiter output signal (pu).
        # ierr[46] = psspy.chsb(0, 1, [-1, -1, -1, 1, 24, 0])  # VOEL, maximum excitation limiter output signal (pu).
        ierr[47] = psspy.chsb(0, 1, [-1, -1, -1, 1, 25, 0])  # PLOAD.
        ierr[48] = psspy.chsb(0, 1, [-1, -1, -1, 1, 26, 0])  # QLOAD.
        # ierr[49] = psspy.chsb(0, 1, [-1, -1, -1, 1, 27, 0])  # GREF, turbine governor reference.
        # ierr[50] = psspy.chsb(0, 1, [-1, -1, -1, 1, 28, 0])  # LCREF, turbine load control reference.
        # ierr[51] = psspy.chsb(0, 1, [-1, -1, -1, 1, 29, 0])  # WVLCTY, wind velocity (m/s).
        # ierr[52] = psspy.chsb(0, 1, [-1, -1, -1, 1, 30, 0])  # WTRBSP, wind turbine rotor speed deviation (pu).
        # ierr[53] = psspy.chsb(0, 1, [-1, -1, -1, 1, 31, 0])  # WPITCH, pitch angle (degrees).
        # ierr[54] = psspy.chsb(0, 1, [-1, -1, -1, 1, 32, 0])  # WAEROT, aerodynamic torque (pu on MBASE).
        # ierr[55] = psspy.chsb(0, 1, [-1, -1, -1, 1, 33, 0])  # WROTRV, rotor voltage (pu on MBASE).
        # ierr[56] = psspy.chsb(0, 1, [-1, -1, -1, 1, 34, 0])  # WROTRI, rotor current (pu on MBASE).
        # ierr[57] = psspy.chsb(0, 1, [-1, -1, -1, 1, 35, 0])  # WPCMND, active power command from wind control (pu on MBASE).
        # ierr[58] = psspy.chsb(0, 1, [-1, -1, -1, 1, 36, 0])  # 36 WQCMND, reactive power command from wind control (pu on MBASE).
        # ierr[59] = psspy.chsb(0, 1, [-1, -1, -1, 1, 37, 0])  # WAUXSG, output of wind auxiliary control (pu on MBASE).
    elif co == 'GVEA':
        ierr[60] = psspy.bsys(1, 0, [0.0, 0.0], 0, [], 9, [325, 33601, 33500, 30110, 30120, 31001, 32001, 32300, 32100],
                              0, [], 0, [])
        ierr[61] = psspy.chsb(1, 0, [-1, -1, -1, 1, 2, 0])  # Machine electrical power
        ierr[62] = psspy.chsb(1, 0, [-1, -1, -1, 1, 12, 0])  # Bus Frequency Deviations (pu)
        ierr[63] = psspy.chsb(1, 0, [-1, -1, -1, 1, 13, 0])  # Bus Voltage and angle (pu)
    print(ierr)
    ierr = [1] * 10
    ierr[1] = psspy.strt(0, os.path.join(basepath, out))  # (psse33) Initialize dynamic simulation
    ierr[2] = psspy.run(0, tst, 0, 1, 1)  # 设置有功负荷突变时刻
    load_real = (1 + float(ld) / float(100)) * lodcplx.real  # 突变后负荷值
    z_ratio, i_ratio, p_ratio = float(lz_[0]) / 100, float(lz_[1]) / 100, float(lz_[2]) / 100  # 按比例分配
    ierr[3] = psspy.load_data_4(lodbus, lodid, [_i, _i, _i, _i, _i, _i],
                                # PL/QL/IP/IQ/YP/YQ
                                [load_real * p_ratio, _f, load_real * i_ratio, _f, load_real * z_ratio, _f])
    # ierr[4] = psspy.run(0, tet, 0, 1, 1)  # 设置故障结束时刻
    # ierr[5] = psspy.dist_clear_fault(1)  # clears fault
    ierr[6] = psspy.run(0, rt, 0, 1, 1)  # 设置仿真结束时间
    ierr[7] = psspy.delete_all_plot_channels()  # Delete plot channels to get ready for next simulation
    time.sleep(0.1)
    console_output = tail(log_path, 10)
    console_output = ''.join(console_output)
    print(ierr)
    # Check for errors written in output file
    current_error = 0
    if "Network not converged" in console_output:
        print('Network not converged')
        current_error = 1
    # raise SystemExit #this will quit the program, if the program is called within a larger
    # program, like optimization, you will want to stop PSSe from running using this or
    # have this rerun the program or skip this iteration's results
    elif "NaN" in console_output:
        print("NaN, network is no good")
        current_error = 1
    # raise SystemExit #this will quit the program, if the program is called within a larger
    # program, like optimization, you will want to stop PSSe from running using this or
    # have this rerun the program or skip this iteration's results
    if current_error == 0 and "INITIAL CONDITIONS CHECK O.K." in console_output:
        print("No errors and initial conditions were good.")

    if current_error > 0:
        return

    # Gather the data and output to excel
    data = dyntools.CHNF(os.path.join(basepath, out))  # getting data from channel.out file
    d, e, z = data.get_data()  # gathering data from data in dictionary format
    # z contains the time series output data from each channel
    # e contains the label of each channel
    # d contains the header of the output file (case name, etc.)
    sim_re = fetch_results(d, e, z)
    xlsx = os.path.join(basepath, os.path.join(basepath, out)[:-4] + '.xlsx')
    # Save Data to excel file
    with pd.ExcelWriter(xlsx) as writer:
        for k, v in sim_re.items():
            v.to_excel(writer, sheet_name=k)
    return


def cut_machine(basepath, out, tst, tet, gbus, gid, rt, sav, df, sp, co, re_machi_id):
    le_, lz_, rr_, hi_ = sp['le'], sp['lz'], sp['rr'], sp['hi']
    ierr = [1] * 100  # check and record for error codes
    ierr[1] = psspy.case(sav)  # load case information (.sav file)

    """ 运行方式设置 """
    # 读取系统当前总负荷和总发电量
    err_load, demand = psspy.systot('LOAD')  # Total system load.
    err_gen, supply = psspy.systot('GEN')  # Total system generation
    demand_mw, supply_mw = demand.real, supply.real  # 系统总负荷, 系统总有功出力.
    try:
        assert supply_mw >= demand_mw, "total_generation < total_load !"
    except Exception as e:
        print(e)
    ierr[2], ierr[3] = err_load, err_gen

    # 读取系统当前各负荷分量和各发电量分量
    ierr[4], (genbuses,) = psspy.amachint(-1, 1, 'NUMBER')
    ierr[5], (genid,) = psspy.amachchar(-1, 1, 'ID')
    # ierr[6], (genpq,) = psspy.amachcplx(-1, 1, 'PQGEN')  # 'O_PQGEN', 该api也可提取发电机参数
    ierr[6], (genp,) = psspy.amachreal(-1, 1, 'PGEN')
    ierr[7], (genpmax,) = psspy.amachreal(-1, 1, 'PMAX')
    generators = zip(genbuses, genid, genp, genpmax)

    ierr[8], (loadbuses,) = psspy.aloadint(-1, 1, 'NUMBER')
    ierr[9], (loadid,) = psspy.aloadchar(-1, 1, 'ID')
    ierr[10], (loadt,) = psspy.aloadcplx(-1, 1, 'TOTALACT')  # TOTALACT, TOTALNOM
    loadp = [lt.real for lt in loadt]
    try:
        assert round(demand_mw, 3) == round(sum(loadp), 3), "demand_mw != sum(loadp) !"
    except Exception as e:
        print(e)
    loads = zip(loadbuses, loadid, loadp)

    # 负荷水平设置(通过设置IPload)
    # 将此轮负荷水平变动按照原始各负荷分量占比分配
    ierr[11], (loadip,) = psspy.aloadcplx(-1, 1, 'TOTALNOM')  # TOTALACT, TOTALNOM
    ipload = [lp.real for lp in loadip]
    load_ip = zip(loadbuses, ipload)
    for l, i in load_ip:
        err = psspy.load_chng_4(l, r"""1""", [_i, _i, _i, _i, _i, _i], [_f, _f, i * le_ / 100, _f, _f, _f])
        assert err == 0, "load change fault !"

    # 旋转备用设置(修改发电机Pmax)
    # 将此轮旋转备用变动按照原始各发电量分量占比分配
    total_rr = demand_mw * rr_  # 系统总的旋转备用
    for b, i, p, pm in generators:
        err = psspy.machine_chng_2(b, r"""1""", [_i, _i, _i, _i, _i, _i],
                                   [_f, _f, _f, _f, p + (p / sum(genp)) * total_rr,
                                    _f, _f, _f, _f, _f, _f, _f, _f, _f, _f, _f, _f])
        assert err == 0, "machine change fault !"

    # 潮流不稳定直接跳过
    ierr[12] = psspy.fnsl([0, 0, 0, 1, 1, 0, 99, 0])
    # ierr[12] = psspy.fdns([0, 0, 0, 1, 1, 0, 99, 0])  # fixed slope decoupled Newton-Raphson power flow calculation
    # assert ierr[12] == 0, "Error in power flow calculation! \n %s" % out_file
    if ierr[12] != 0:
        print("%s \npower flow is not converged!" % out)
        return

    # 重跑潮流后线路及其运行数据
    loadbuses_, demand_mw_, generators_, genp_, loads_, branch_flows = get_system_info()

    # 负荷模型设置
    ierr[13] = psspy.cong(0)  # Convert Generators
    ierr[14] = psspy.conl(0, 1, 1, [0, 0], [lz_[1], lz_[0], lz_[1], lz_[0]])  # initialize for load conversion
    ierr[15] = psspy.conl(0, 1, 2, [0, 0], [lz_[1], lz_[0], lz_[1], lz_[0]])  # convert loads; 'PI', 'PZ', 'QI', 'QZ'
    ierr[16] = psspy.conl(0, 1, 3, [0, 0], [lz_[1], lz_[0], lz_[1], lz_[0]])  # postprocessing housekeeping
    ierr[17] = psspy.ordr(1)
    ierr[18] = psspy.fact()
    ierr[19] = psspy.tysl(0)
    ierr[20] = psspy.dyre_new([1, 1, 1, 1], df, "", "", "")

    # 惯性时间常数设置(GENROU:30-38, GENCLS:39)
    h_GENROU, nu = [6.05, 3.41, 6.05, 3.41, 5.016, 3.141, 3.141, 5.32], [30, 31, 32, 34, 35, 36, 37, 38]
    h_GENCLS, ns = [500], [39]
    for u in range(len(nu)):
        err = psspy.change_plmod_con(nu[u], r"""1""", r"""GENROU""", 5, h_GENROU[u] * hi_)
        assert err == 0, "GENROU h change fault! Bus num %s" % nu[u]
    for s in range(len(ns)):
        err = psspy.change_plmod_con(ns[s], r"""1""", r"""GENCLS""", 1, h_GENCLS[s] * hi_)
        assert err == 0, "GENCLS h change fault! Bus num %s" % ns[s]

    """ 扰动故障前置 """
    ierr[22] = psspy.set_relang(1, re_machi_id, r"""1""")  # 设置功角相对值, use machine ID(30) as the reference machine

    # Pick Channels
    if co == 'All':  # 2,3,6,12,13,14,15,16,25,26
        # ierr[23] = psspy.chsb(0, 1, [-1, -1, -1, 1, 1, 0])  # ANGLE, machine relative rotor angle (degrees).
        ierr[24] = psspy.chsb(0, 1, [-1, -1, -1, 1, 2, 0])  # PELEC, machine electrical power (pu on SBASE).
        ierr[25] = psspy.chsb(0, 1, [-1, -1, -1, 1, 3, 0])  # QELEC, machine reactive power.
        # ierr[26] = psspy.chsb(0, 1, [-1, -1, -1, 1, 4, 0])  # ETERM, machine terminal voltage (pu).
        # ierr[27] = psspy.chsb(0, 1, [-1, -1, -1, 1, 5, 0])  # EFD, generator main field voltage (pu).
        ierr[28] = psspy.chsb(0, 1, [-1, -1, -1, 1, 6, 0])  # PMECH, turbine mechanical power (pu on MBASE).
        # ierr[29] = psspy.chsb(0, 1, [-1, -1, -1, 1, 7, 0])  # SPEED, machine speed deviation from nominal (pu).
        # ierr[30] = psspy.chsb(0, 1, [-1, -1, -1, 1, 8, 0])  # XADIFD, machine field current (pu).
        # ierr[31] = psspy.chsb(0, 1, [-1, -1, -1, 1, 9, 0])  # ECOMP, voltage regulator compensated voltage (pu).
        # ierr[32] = psspy.chsb(0, 1, [-1, -1, -1, 1, 10, 0])  # VOTHSG, stabilizer output signal (pu).
        # ierr[33] = psspy.chsb(0, 1, [-1, -1, -1, 1, 11, 0])  # VREF, voltage regulator voltage setpoint (pu).
        ierr[34] = psspy.chsb(0, 1, [-1, -1, -1, 1, 12, 0])  # BSFREQ, bus pu frequency deviations.
        ierr[35] = psspy.chsb(0, 1, [-1, -1, -1, 1, 13, 0])  # VOLT, bus pu voltages (complex).
        ierr[36] = psspy.chsb(0, 1, [-1, -1, -1, 1, 14, 0])  # voltage and angle
        ierr[37] = psspy.chsb(0, 1, [-1, -1, -1, 1, 15, 0])  # flow (P).
        ierr[38] = psspy.chsb(0, 1, [-1, -1, -1, 1, 16, 0])  # flow (P and Q).
        # ierr[39] = psspy.chsb(0, 1, [-1, -1, -1, 1, 17, 0])  # flow (MVA).
        # ierr[40] = psspy.chsb(0, 1, [-1, -1, -1, 1, 18, 0])  # apparent impedance (R and X).
        # ierr[43] = psspy.chsb(0, 1, [-1, -1, -1, 1, 21, 0])  # ITERM.
        # ierr[44] = psspy.chsb(0, 1, [-1, -1, -1, 1, 22, 0])  # machine apparent impedance
        # ierr[45] = psspy.chsb(0, 1, [-1, -1, -1, 1, 23, 0])  # VUEL, minimum excitation limiter output signal (pu).
        # ierr[46] = psspy.chsb(0, 1, [-1, -1, -1, 1, 24, 0])  # VOEL, maximum excitation limiter output signal (pu).
        ierr[47] = psspy.chsb(0, 1, [-1, -1, -1, 1, 25, 0])  # PLOAD.
        ierr[48] = psspy.chsb(0, 1, [-1, -1, -1, 1, 26, 0])  # QLOAD.
        # ierr[49] = psspy.chsb(0, 1, [-1, -1, -1, 1, 27, 0])  # GREF, turbine governor reference.
        # ierr[50] = psspy.chsb(0, 1, [-1, -1, -1, 1, 28, 0])  # LCREF, turbine load control reference.
        # ierr[51] = psspy.chsb(0, 1, [-1, -1, -1, 1, 29, 0])  # WVLCTY, wind velocity (m/s).
        # ierr[52] = psspy.chsb(0, 1, [-1, -1, -1, 1, 30, 0])  # WTRBSP, wind turbine rotor speed deviation (pu).
        # ierr[53] = psspy.chsb(0, 1, [-1, -1, -1, 1, 31, 0])  # WPITCH, pitch angle (degrees).
        # ierr[54] = psspy.chsb(0, 1, [-1, -1, -1, 1, 32, 0])  # WAEROT, aerodynamic torque (pu on MBASE).
        # ierr[55] = psspy.chsb(0, 1, [-1, -1, -1, 1, 33, 0])  # WROTRV, rotor voltage (pu on MBASE).
        # ierr[56] = psspy.chsb(0, 1, [-1, -1, -1, 1, 34, 0])  # WROTRI, rotor current (pu on MBASE).
        # ierr[57] = psspy.chsb(0, 1, [-1, -1, -1, 1, 35, 0])  # WPCMND, active power command from wind control (pu on MBASE).
        # ierr[58] = psspy.chsb(0, 1, [-1, -1, -1, 1, 36, 0])  # 36 WQCMND, reactive power command from wind control (pu on MBASE).
        # ierr[59] = psspy.chsb(0, 1, [-1, -1, -1, 1, 37, 0])  # WAUXSG, output of wind auxiliary control (pu on MBASE).
    elif co == 'GVEA':
        ierr[60] = psspy.bsys(1, 0, [0.0, 0.0], 0, [], 9, [325, 33601, 33500, 30110, 30120, 31001, 32001, 32300, 32100],
                              0, [], 0, [])
        ierr[61] = psspy.chsb(1, 0, [-1, -1, -1, 1, 2, 0])  # Machine electrical power
        ierr[62] = psspy.chsb(1, 0, [-1, -1, -1, 1, 12, 0])  # Bus Frequency Deviations (pu)
        ierr[63] = psspy.chsb(1, 0, [-1, -1, -1, 1, 13, 0])  # Bus Voltage and angle (pu)
    print(ierr)
    ierr = [1] * 10
    ierr[1] = psspy.strt(0, os.path.join(basepath, out))  # (psse33) Initialize dynamic simulation
    ierr[2] = psspy.run(0, tst, 0, 1, 1)  # 设置故障触发时刻
    ierr[3] = psspy.dist_machine_trip(gbus)
    ierr[4] = psspy.run(0, tet, 0, 1, 1)  # 设置故障结束时刻
    ierr[5] = psspy.dist_clear_fault(1)  # clears fault
    ierr[6] = psspy.run(0, rt, 0, 1, 1)  # 设置仿真结束时间
    ierr[7] = psspy.delete_all_plot_channels()  # Delete plot channels to get ready for next simulation
    time.sleep(0.1)
    console_output = tail(log_path, 10)
    console_output = ''.join(console_output)
    print(ierr)
    # Check for errors written in output file
    current_error = 0
    if "Network not converged" in console_output:
        print('Network not converged')
        current_error = 1
    # raise SystemExit #this will quit the program, if the program is called within a larger
    # program, like optimization, you will want to stop PSSe from running using this or
    # have this rerun the program or skip this iteration's results
    elif "NaN" in console_output:
        print("NaN, network is no good")
        current_error = 1
    # raise SystemExit #this will quit the program, if the program is called within a larger
    # program, like optimization, you will want to stop PSSe from running using this or
    # have this rerun the program or skip this iteration's results
    if current_error == 0 and "INITIAL CONDITIONS CHECK O.K." in console_output:
        print("No errors and initial conditions were good.")

    if current_error > 0:
        return

    # Gather the data and output to excel
    data = dyntools.CHNF(os.path.join(basepath, out))  # getting data from channel.out file
    d, e, z = data.get_data()  # gathering data from data in dictionary format
    # z contains the time series output data from each channel
    # e contains the label of each channel
    # d contains the header of the output file (case name, etc.)
    sim_re = fetch_results(d, e, z)
    xlsx = os.path.join(basepath, os.path.join(basepath, out)[:-4] + '.xlsx')
    # Save Data to excel file
    with pd.ExcelWriter(xlsx) as writer:
        for k, v in sim_re.items():
            v.to_excel(writer, sheet_name=k)
    return


# Collect and Sort PSSe Output Data
def fetch_results(d, e, z):
    # This function takes the outputs from PSSe and assigns the data to pandas dataframes
    # according to data type. This function will need to be updated if additional
    # channels are added.
    # Initialize dataframes for each output type
    total_keys = []  # 提取e中的所有有效key
    total_df = {}  # 将每个key对应的数据存为dataframe

    # Sort by channel type (machine electric power, bus frequency deviation, bus voltage angle, bus voltage)
    # Append dataframe with channel data, and increase index for inserting next channel's data
    for channel in range(1, len(e)):  # Check length of 'e' and run for all those channels
        channel_keys = re.split(' |\[|\]', e[channel])  # Parse name of channel
        total_keys.append(channel_keys[0])
    total_keys = list(set(total_keys))

    # 初始化total_df
    for tk in total_keys:
        total_df[tk] = pd.DataFrame()

    for channel in range(1, len(e)):  # Check length of 'e' and run for all those channels
        channel_keys = re.split(' |\[|\]', e[channel])  # Parse name of channel
        tmp_key = channel_keys[0]
        if len(total_df[tmp_key]) == 0:
            total_df[tmp_key] = pd.DataFrame(z[channel], columns=[channel_keys[1]], index=z['time'])
        else:
            tmp_df = copy.deepcopy(total_df[tmp_key])
            tmp_df.insert(tmp_df.shape[1], channel_keys[1], z[channel], allow_duplicates=True)
            total_df[tmp_key] = tmp_df

    return total_df


def tail(filepath, n, block=-1024):
    with open(filepath, 'rb') as f:
        f.seek(0, 2)
        filesize = f.tell()
        while True:
            if filesize >= abs(block):
                f.seek(block, 2)
                s = f.readlines()
                if len(s) > n:
                    return s[-n:]
                    break
                else:
                    block *= 2
            else:
                block = -filesize


# Run Dynamic Simulation, Collect Data
if __name__ == "__main__":
    err1 = psspy.psseinit(200000)  # initialize PSSe.
    assert err1 == 0, "PSSe initialization failed"
    log_path = './logs/log.txt' # 输出日志文件
    err2 = psspy.progress_output(2, log_path)
    err3 = psspy.alert_output(2, log_path)
    err4 = psspy.report_output(2, log_path)
    err5 = psspy.prompt_output(2, log_path)

    """ useful files and settings """
    sav_file_name = 'IEEE39'
    sav_file = './Demo_Models/IEEE39/%s.sav' % sav_file_name
    dyr_file = './Demo_Models/IEEE39/IEEE39bus_rew_10_v33.dyr'

    trigger_start_time = 1.0
    trigger_end_time = 1.17
    runtime = 30  # length of simulation run
    left_limit = 0  # left axis-limit for plots
    right_limit = 30  # right axis-limit for plots
    channel_option = 'All'  # Create output channels for all machines and buses

    """ 运行方式定义 """
    # load_level = np.arange(50, 110.25, 2.5)  # 25种负荷水平设置%
    # # 5种负荷模型zip比例设置%
    # load_zip = [[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [40.0, 0.0, 60.0], [35.0, 35.0, 30.0], [30.0, 40.0, 30.0]]
    # reserve_ratio = np.arange(0, 4.5, 0.5)  # 9种旋转备用容量比例%（相对于总负荷）设置
    # h_inertia = np.arange(0.2, 2.2, 0.2)  # 10种惯性时间常数比例设置

    load_level = [100.0, 90.0, 110.0, 80.0, 120.0, 70.0, 130.0]  # 7种负荷水平设置%
    load_zip = [[40.0, 0.0, 60.0], [100.0, 0.0, 0.0], [30.0, 40.0, 30.0]]  # 3种负荷模型zip比例设置%
    reserve_ratio = [0.0, 2.0, 4.0]  # 3种旋转备用容量比例%（相对于总负荷）设置
    h_inertia = [0.2, 0.5, 1.0, 2.0]  # 4种惯性时间常数比例设置

    """ 扰动类型定义 """
    disturbance_type = ["cut_machine", "cut_machine_", "load_change"]
    load_delta = [-300.0, 300.0, -200.0, 200.0, -10.0, 10.0]  # 负荷突变水平%

    """ start simulation """
    for le in tqdm(load_level):
        for lz in load_zip:
            for rr in reserve_ratio:
                for hi in h_inertia:
                    # 当前运行方式
                    t0 = time.time()
                    steady_param = dict(zip(['le', 'lz', 'rr', 'hi'], [le, lz, rr, hi]))
                    loadbuses, demand_mw, generators, genp, loads, branch_flows = get_system_info(sav_file)

                    # """ cut_machine """
                    # for gbus, gid, gp, gpmax in generators:
                    #     # saved files names' setting
                    #     path_base = './Outputs/IEEE39/cut_machine'
                    #     case_name = '%s_%s%s_%s%s-%s-%s_%s%s_%s%s-cut_machine-%s_%s-%s_%ss' % \
                    #                 (sav_file_name, 'le', le, 'zip', lz[0], lz[1], lz[2],
                    #                  'rr', rr, 'hi', hi, 'gbus', gbus, channel_option, runtime)
                    #     print(case_name)
                    #     out_file = 'psse3304_%s.out' % case_name
                    #     if os.path.exists(os.path.join(path_base, out_file)):
                    #         continue
                    #     cut_machine(path_base, out_file, trigger_start_time, trigger_end_time, gbus, gid, runtime,
                    #                 sav_file, dyr_file, steady_param, channel_option, 30)

                    # """ load_change """
                    # for ld in load_delta:
                    #     for lodbus, lodid, lodcplx in loads:
                    #         # saved files names' setting
                    #         path_base = './Outputs/IEEE39/load_change'
                    #         case_name = '%s_%s%s_%s%s-%s-%s_%s%s_%s%s_%s_%s-load_change-%s_%s-%s_%ss' % \
                    #                     (sav_file_name, 'le', le, 'zip', lz[0], lz[1], lz[2],
                    #                      'rr', rr, 'hi', hi, 'ld', ld, 'lodbus', lodbus, channel_option, runtime)
                    #         print(case_name)
                    #         out_file = 'psse3304_%s.out' % case_name
                    #         if os.path.exists(os.path.join(path_base, out_file)):
                    #             continue
                    #
                    #         try:
                    #             result = func_timeout(120, load_change, args=(path_base, out_file,
                    #                                                         trigger_start_time, trigger_end_time,
                    #                                                         ld, lodbus, lodid, lodcplx, runtime,
                    #                                                         sav_file, dyr_file, steady_param,
                    #                                                         channel_option, 30))
                    #             print(result)
                    #         except FunctionTimedOut:
                    #             print("Function call timed out")

                    """ circuit_short """
                    for frmbus, tobus, cktid, pflow in  branch_flows:
                        # saved files names' setting
                        path_base = './Outputs/IEEE39/circuit_short'
                        case_name = '%s_%s%s_%s%s-%s-%s_%s%s_%s%s-circuit_short-%s%s_%s%s-%s_%ss' % \
                                    (sav_file_name, 'le', le, 'zip', lz[0], lz[1], lz[2],
                                     'rr', rr, 'hi', hi, 'frmbus', frmbus, 'tobus', tobus, channel_option, runtime)
                        print(case_name)
                        out_file = 'psse3304_%s.out' % case_name
                        if os.path.exists(os.path.join(path_base, out_file)):
                            continue
                        circuit_short(path_base, out_file, trigger_start_time, trigger_end_time, frmbus, tobus, cktid,
                                      runtime, sav_file, dyr_file, steady_param, channel_option, 30)
                        pass
