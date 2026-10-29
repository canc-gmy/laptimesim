"""Generates standalone visual map and subsystem characteristic charts for the PDF report:

1. Pacejka MF5.2 lateral force Fy vs. slip angle alpha curves at multiple normal loads (500N, 1000N, 1500N, 2000N)
2. Pacejka longitudinal force Fx vs. slip ratio kappa curves
3. 2D Aerodynamic Map contours / surface: Cl and Cd vs. front & rear ride height
4. Electric motor torque-speed and power envelope curve (240 Nm cap, 80 kW ceiling up to 6500 RPM)
5. Nonlinear suspension force vs. deflection curve showing linear travel and progressive bump stop engagement
6. 4-corner tire normal load distribution (Fz_FL, Fz_FR, Fz_RL, Fz_RR) along track distance s
7. Tire friction circle utilization (% of max grip used) along track distance s
"""

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def generate_all_subsystem_map_charts(output_dir: str = "results_fsg_benchmark/maps") -> dict:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    generated_files = {}

    # 1. Pacejka Tire Lateral Force Curves
    fig, ax = plt.subplots(figsize=(6, 4))
    alpha_deg = np.linspace(-15, 15, 300)
    alpha_rad = np.deg2rad(alpha_deg)
    B, C, D, E = 10.5, 1.65, 1.55, -0.4

    for Fz in [500, 1000, 1500, 2000]:
        bx = B * alpha_rad
        inner = bx - E * (bx - np.arctan(bx))
        Fy = D * Fz * np.sin(C * np.arctan(inner))
        ax.plot(alpha_deg, Fy, label=f"$F_z$ = {Fz} N", linewidth=1.5)

    ax.set_title("Pacejka MF 5.2: Lateral Force $F_y$ vs. Slip Angle $\\alpha$")
    ax.set_xlabel("Slip Angle $\\alpha$ [deg]")
    ax.set_ylabel("Lateral Force $F_y$ [N]")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower right")
    f_tire_lat = out_path / "map_tire_lateral.png"
    fig.savefig(f_tire_lat, dpi=200, bbox_inches="tight")
    plt.close(fig)
    generated_files["tire_lateral"] = f_tire_lat

    # 2. Pacejka Tire Longitudinal Force Curves
    fig, ax = plt.subplots(figsize=(6, 4))
    kappa = np.linspace(-0.25, 0.25, 300)
    B_long, C_long, D_long, E_long = 11.0, 1.68, 1.60, -0.4

    for Fz in [500, 1000, 1500, 2000]:
        bx = B_long * kappa
        inner = bx - E_long * (bx - np.arctan(bx))
        Fx = D_long * Fz * np.sin(C_long * np.arctan(inner))
        ax.plot(kappa * 100.0, Fx, label=f"$F_z$ = {Fz} N", linewidth=1.5)

    ax.set_title("Pacejka MF 5.2: Longitudinal Traction $F_x$ vs. Slip Ratio $\\kappa$")
    ax.set_xlabel("Longitudinal Slip Ratio $\\kappa$ [%]")
    ax.set_ylabel("Longitudinal Force $F_x$ [N]")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower right")
    f_tire_long = out_path / "map_tire_longitudinal.png"
    fig.savefig(f_tire_long, dpi=200, bbox_inches="tight")
    plt.close(fig)
    generated_files["tire_longitudinal"] = f_tire_long

    # 3. 2D Aerodynamic Downforce (Cl) Map Contour
    fig, ax = plt.subplots(figsize=(6, 4.5))
    rh_f_grid = np.array([20, 30, 40, 50])  # mm
    rh_r_grid = np.array([25, 35, 45, 55])  # mm
    cl_table = np.array([
        [3.4, 3.3, 3.1, 2.9],
        [3.5, 3.4, 3.2, 3.0],
        [3.3, 3.2, 3.0, 2.8],
        [3.1, 3.0, 2.8, 2.6],
    ])
    # Interpolate high-res surface
    from scipy.interpolate import RectBivariateSpline
    spline = RectBivariateSpline(rh_f_grid, rh_r_grid, cl_table)
    x_fine = np.linspace(20, 50, 100)
    y_fine = np.linspace(25, 55, 100)
    Z_fine = spline(x_fine, y_fine)

    c = ax.contourf(x_fine, y_fine, Z_fine.T, levels=14, cmap="viridis")
    cbar = plt.colorbar(c, ax=ax)
    cbar.set_label("Total Downforce Coefficient $C_l$ [-]")
    ax.set_xlabel("Front Ride Height [mm]")
    ax.set_ylabel("Rear Ride Height [mm]")
    ax.set_title("Aerodynamic Downforce Map ($C_l$ vs. Ride Heights)")
    f_aero_map = out_path / "map_aero_cl_contour.png"
    fig.savefig(f_aero_map, dpi=200, bbox_inches="tight")
    plt.close(fig)
    generated_files["aero_map"] = f_aero_map

    # 4. Electric Motor Torque-Speed & Power Envelope
    fig, ax1 = plt.subplots(figsize=(6, 4))
    rpm = np.linspace(0, 6500, 300)
    omega = rpm * (2 * np.pi / 60.0)
    t_peak = 240.0
    p_peak = 80000.0  # 80 kW ceiling

    torque = np.minimum(t_peak, p_peak / np.maximum(omega, 1e-3))
    power_kw = (torque * omega) / 1000.0

    color1 = "tab:blue"
    ax1.plot(rpm, torque, color=color1, linewidth=1.8, label="Motor Torque [Nm]")
    ax1.set_xlabel("Motor Speed [RPM]")
    ax1.set_ylabel("Torque [Nm]", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_ylim(0, 280)
    ax1.grid(True, linestyle=":", alpha=0.5)

    ax2 = ax1.twinx()
    color2 = "tab:red"
    ax2.plot(rpm, power_kw, color=color2, linewidth=1.8, linestyle="--", label="Power [kW]")
    ax2.set_ylabel("Electrical / Mech Power [kW]", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.axhline(80.0, color="darkred", linestyle=":", label="80 kW FSG Rule Cap")
    ax2.set_ylim(0, 95)

    ax1.set_title("Electric Powertrain Torque-Speed & 80 kW Power Limit")
    f_motor = out_path / "map_motor_envelope.png"
    fig.savefig(f_motor, dpi=200, bbox_inches="tight")
    plt.close(fig)
    generated_files["motor_envelope"] = f_motor

    # 5. Nonlinear Suspension & Bump Stop Curve
    fig, ax = plt.subplots(figsize=(6, 4))
    deflection_mm = np.linspace(-15, 30, 300)  # rebound to compression
    deflection_m = deflection_mm / 1000.0
    k_main = 38000.0  # N/m
    k_bump = 120000.0
    clearance_m = 0.015  # 15 mm bump clearance

    f_linear = k_main * deflection_m
    bump_comp = np.maximum(0.0, deflection_m - clearance_m)
    f_bump = k_bump * (bump_comp**1.5)
    f_total = f_linear + f_bump

    ax.plot(deflection_mm, f_linear, "k--", label="Main Spring Only", linewidth=1.2)
    ax.plot(deflection_mm, f_total, "crimson", label="Total Suspension (Spring + Bump Stop)", linewidth=1.8)
    ax.axvline(15.0, color="gray", linestyle=":", label="Bump Stop Contact (15 mm)")
    ax.set_title("Suspension Nonlinear Spring & Progressive Bump Stop Curve")
    ax.set_xlabel("Suspension Travel (Compression > 0) [mm]")
    ax.set_ylabel("Vertical Reaction Force [N]")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left")
    f_susp = out_path / "map_suspension_curve.png"
    fig.savefig(f_susp, dpi=200, bbox_inches="tight")
    plt.close(fig)
    generated_files["suspension_curve"] = f_susp

    # 6. Four-Corner Normal Load Transfer Traces from Telemetry
    telem_csv = Path("results_fsg_benchmark/telemetry.csv")
    if telem_csv.is_file():
        df = pd.read_csv(telem_csv)
        s = df["s_m"]

        fig, ax = plt.subplots(figsize=(7, 3.8))
        if "telem_Fz_fl" in df and "telem_Fz_fr" in df:
            ax.plot(s, df["telem_Fz_fl"], color="blue", linewidth=1.3, label="FL (Front Left)")
            ax.plot(s, df["telem_Fz_fr"], color="cyan", linewidth=1.3, label="FR (Front Right)")
            ax.plot(s, df["telem_Fz_rl"], color="red", linewidth=1.3, label="RL (Rear Left)")
            ax.plot(s, df["telem_Fz_rr"], color="orange", linewidth=1.3, label="RR (Rear Right)")
            ax.set_title("Dynamic Normal Wheel Loads ($F_{z}$) Along Lap")
            ax.set_xlabel("Track Distance $s$ [m]")
            ax.set_ylabel("Vertical Load $F_z$ [N]")
            ax.grid(True, linestyle=":", alpha=0.6)
            ax.legend(loc="upper right", ncol=2)
            f_loads = out_path / "telemetry_wheel_loads.png"
            fig.savefig(f_loads, dpi=200, bbox_inches="tight")
            plt.close(fig)
            generated_files["wheel_loads"] = f_loads

        # 7. Friction Circle Utilization Profile
        fig, ax = plt.subplots(figsize=(7, 3.8))
        if "ax_g" in df and "ay_g" in df:
            g_total = np.hypot(df["ax_g"], df["ay_g"])
            # Estimate peak grip capacity (~3.6g at high speed downforce, ~1.8g at low speed)
            speed_ratio = df["speed_km_h"] / df["speed_km_h"].max()
            g_limit = 1.8 + 1.8 * (speed_ratio**2)
            utilization_pct = np.clip((g_total / g_limit) * 100.0, 10.0, 100.0)

            ax.plot(s, utilization_pct, color="darkmagenta", linewidth=1.5, label="Tire Friction Circle Utilization [%]")
            ax.axhline(100.0, color="red", linestyle="--", label="100% Grip Limit")
            ax.set_title("Vehicle Dynamic Grip Utilization Along Circuit")
            ax.set_xlabel("Track Distance $s$ [m]")
            ax.set_ylabel("Friction Utilization [%]")
            ax.set_ylim(0, 110)
            ax.grid(True, linestyle=":", alpha=0.6)
            ax.legend(loc="lower right")
            f_util = out_path / "telemetry_grip_utilization.png"
            fig.savefig(f_util, dpi=200, bbox_inches="tight")
            plt.close(fig)
            generated_files["grip_utilization"] = f_util

    return generated_files


if __name__ == "__main__":
    generate_all_subsystem_map_charts()
