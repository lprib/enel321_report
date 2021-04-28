#!/usr/bin/env python

import matplotlib.pyplot as pp
import matplotlib.patches as mpatches
import csv
import matplotlib
import numpy as np
import scipy.signal as sig

matplotlib.use("pgf")

matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})


def generate_data_dict(file_name):
    data = {}
    with open(file_name) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (float(row["time"]) < 19.9) and (float(row["time"]) > 9.0):
                for key in row.keys():
                    if key in data:
                        data[key].append(float(row[key]))
                    else:
                        data[key] = []
    return data


def do_plot(ax, time_data, plots_data, xlabel, ylabel):
    colors = ["#F62619", "#332771", "#F9AD09", "#880040"]
    for (i, plot_data) in enumerate(plots_data):
        ax.plot(time_data, plot_data, color=colors[i], linewidth=1)
    pp.xlabel(xlabel)
    pp.ylabel(ylabel)
    pp.grid(True)


# data = generate_data_dict("./data/3.csv")
# do_plot(data["time"], [data["roll_angle"], data["setpoint"]], [
    # "Roll Angle (degrees)", "Commanded Angle"], "Time (s)", "Angle (degrees)", "graph1.pgf")


def settle_time(time_arr, response_arr):
    threshold = 0.02
    last_bad_idx = 0
    for (i, r) in enumerate(response_arr):
        if (r < 90-90.0*threshold) or (r > 90+90.0*threshold):
            last_bad_idx = i
    return time_arr[last_bad_idx]


def get_specs(time_arr, response_arr):
    # (tr, Mp, tp, xi, t10, t90, ess, peak, ts)
    time_arr = np.array(time_arr)
    response_arr = np.array(response_arr)
    peak_idx = np.argmax(response_arr)
    peak_resp = response_arr[peak_idx]
    peak_time = time_arr[peak_idx]
    mp_percent = (peak_resp - 90.0) / 90.0 * 100.0
    xi = np.sqrt((np.log(mp_percent/100) ** 2) /
                 (np.pi ** 2 + np.log(mp_percent/100) ** 2))

    t_10p = time_arr[np.argmax(response_arr > (90.0*0.1))]
    t_90p = time_arr[np.argmax(response_arr > (90.0*0.9))]
    tr = t_90p - t_10p

    t19s_idx = np.argmax(time_arr > (19.5))
    ess = response_arr[t19s_idx] - 90.0

    ts = settle_time(time_arr, response_arr)
    return (tr, mp_percent, peak_time, xi, t_10p, t_90p, ess, peak_resp, ts)


def simulate(kp, ki, kd, time_arr, reference_arr, ax):
    a = 0.94
    b = 9
    c = 0.4
    system = sig.lti([0, 0, b*kp, b*ki], [1, a+b*kd, c+b*kp, b*ki])
    time_arr = np.linspace(9, 19.9, 500)
    reference_arr = [90 if x > 10 else 0 for x in time_arr]
    tout, response, _ = sig.lsim(system, reference_arr, time_arr, interp=0)
    return time_arr, response


def gen_response_plot(in_filename, title, out_filename):
    data = generate_data_dict(in_filename)

    tr, mp, tp, xi, t10, t90, ess, peak, ts = get_specs(
        data["time"], data["roll_angle"])

    pp.clf()
    fig, ax = pp.subplots()

    sim_time, sim_response = simulate(float(data["Kp"][0]), float(
        data["Ki"][0]), float(data["Kd"][0]), data["time"], data["setpoint"], ax)
    ax.plot(sim_time, sim_response, color="#F9AD09", linewidth=1)

    do_plot(ax, data["time"], [data["setpoint"],
                               data["roll_angle"]], "Time (seconds)", "Angle (degrees)")
    pp.xticks(np.arange(9, 21, 1.0))
    pp.yticks(np.arange(-10, 120, 10))

    # tr display
    ax.axvspan(t10, t90, alpha=0.2, fc="#F9AD09", ec="#000000")
    ax.axhspan(90 + ess, 90, alpha=0.2, color="#880040", ec="#000000")
    ax.axhspan(90, peak, alpha=0.2, color="#332771", ec="#000000")
    ax.legend(labels=[
        "Simulated angle",
        "Reference angle",
        "Measured angle",
        "Rise time, $t_r = {:.2f}$".format(tr),
        "Steady state error, $e_{{ss}} = {:.2f}$".format(ess),
        "Overshoot, $M_p = {:.2f}\\%$".format(mp)
    ])
    pp.savefig(out_filename)


def gen_response_graphs():
    for i in range(1, 6):
        s = str(i)
        gen_response_plot("./data/{}.csv".format(s),
                          "Controller {}".format(s), "response{}.pgf".format(s))


def gen_commanded_angle_graph(controller):
    data = generate_data_dict("./data/{}.csv".format(str(controller)))
    pp.clf()
    fig, ax = pp.subplots()
    do_plot(ax, data["time"], [data["setpoint"], data["u_cmd"]],
            "Time (seconds)", "Angle (degrees)")
    pp.xticks(np.arange(9, 21, 1.0))
    pp.yticks(np.arange(-20, 110, 10))
    ax.legend(labels=["Reference angle", "Canard angle"])

    pp.savefig("commanded{}.pgf".format(str(controller)))


def gen_commanded_angle_graphs():
    gen_commanded_angle_graph(3)
    gen_commanded_angle_graph(5)


def r(value, decimals=2):
    # round float to decimal places and print to string
    format_str = "{{:.{}f}}".format(decimals)
    return format_str.format(value)


def gen_table():
    with open("./table.txt", "w") as f:
        for i in range(1, 6):
            data = generate_data_dict("./data/{}.csv".format(str(i)))
            tr, mp, tp, xi, t10, t90, ess, peak, ts = get_specs(
                data["time"], data["roll_angle"])
            f.write(" & ".join([
                str(i),
                r(data["Kp"][0], 1),
                r(data["Ki"][0], 1),
                r(data["Kd"][0]),
                r(tr),
                r(tp),
                r(ts),
                r(mp),
                r(ess),
                r(xi)
            ]))
            f.write(" \\\\\n")


gen_commanded_angle_graphs()
gen_response_graphs()


def graph_pid(kp, ki, kd, col):
    a = 0.94
    b = 9
    c = 0.4
    system = sig.lti([0, 0, b*kp, b*ki], [1, a+b*kd, c+b*kp, b*ki])
    time_arr = np.linspace(0, 10, 500)
    reference_arr = np.repeat(90, 500)
    tout, response, _ = sig.lsim(system, reference_arr, time_arr, interp=0)
    pp.plot(time_arr, response, color=(col, 0, 1-col))


def multisim1():
    pp.xlabel("Time (s)")
    pp.ylabel("Angle (degrees)")
    labels = []
    for p in np.arange(0.1, 2.0, 0.3):
        labels.append("$K_p$ = {:.1f}".format(p))
        # print(p)
        graph_pid(p, 0, 0, np.interp(p, [0.1, 2.0], [0.0, 1.0]))
    labels.append("Reference angle")
    time_arr = np.linspace(0, 10, 750)
    reference_arr = np.repeat(90, 750)
    pp.plot(time_arr, reference_arr, color=(0.5, 0.5, 0.5), linewidth=3)
    pp.legend(labels)
    pp.xticks(np.arange(0, 16, 1.0))
    pp.yticks(np.arange(0, 170, 10))
    pp.grid(True)
    pp.savefig("pcontroller_gain.pgf")


def multisim2():
    pp.xlabel("Time (s)")
    pp.ylabel("Angle (degrees)")
    labels = []
    for n in np.arange(0.1, 0.8, 0.1):
        labels.append("$K_d$ = {:.1f}".format(n))
        graph_pid(0.8, 0, n, np.interp(n, [0.1, 0.8], [0.0, 1.0]))
    labels.append("Reference angle")
    time_arr = np.linspace(0, 10, 500)
    reference_arr = np.repeat(90, 500)
    pp.plot(time_arr, reference_arr, color=(0.5, 0.5, 0.5), linewidth=3)
    pp.legend(labels)
    pp.xticks(np.arange(0, 10, 1.0))
    pp.yticks(np.arange(0, 130, 10))
    pp.grid(True)
    pp.savefig("pdcontroller_gain.pgf")


def multisim3():
    pp.xlabel("Time (s)")
    pp.ylabel("Angle (degrees)")
    labels = []
    for n in np.arange(0.05, 0.3, 0.05):
        labels.append("$K_i$ = {:.2f}".format(n))
        graph_pid(0.8, n, 0.3, np.interp(n, [0.05, 0.3], [0.0, 1.0]))
    labels.append("Reference angle")
    time_arr = np.linspace(0, 10, 500)
    reference_arr = np.repeat(90, 500)
    pp.plot(time_arr, reference_arr, color=(0.5, 0.5, 0.5), linewidth=3)
    pp.legend(labels)
    pp.xticks(np.arange(0, 10, 1.0))
    pp.yticks(np.arange(0, 130, 10))
    pp.grid(True)
    pp.savefig("pidcontroller_ki.pgf")


multisim1()
multisim2()
multisim3()
gen_response_graphs()
multisim2()
# pp.show()
