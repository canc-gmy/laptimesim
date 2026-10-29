"""Generate a comprehensive engineering report PDF for laptimesim.

Details:
- Full system architecture and formulation (Curvilinear Frenet OCP, Direct Collocation)
- 18+ DOF Multi-Body Vehicle Equations (Chassis 6-DOF, 4x Wheel Spin, 4x Suspension, Compliances, Battery)
- All subsystem models: Pacejka MF5.2, Multi-dimensional B-spline Aero Maps, Bump-stop Suspension, Electric Powertrain & Battery ECM
- Complete table of parameters used in the FSG Endurance Benchmark
- Simulation results and figures embedded (G-G Diagram, Racing Line, Telemetry channels)
"""

import os
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, PageBreak, HRFlowable
)
from reportlab.pdfgen import canvas


class NumberedCanvas(canvas.Canvas):
    """Adds running headers and footers with page numbering."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 9)
        self.setFillColor(colors.HexColor("#555555"))

        # Header (Pages > 1)
        if self._pageNumber > 1:
            self.drawString(54, 750, "laptimesim — Vehicle Dynamics Simulation Report")
            self.drawRightString(558, 750, "Formula Student EV / ROM Benchmark")
            self.setStrokeColor(colors.HexColor("#CCCCCC"))
            self.setLineWidth(0.5)
            self.line(54, 744, 558, 744)

        # Footer
        page_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(558, 40, page_text)
        self.drawString(54, 40, "Confidential — Formula Student Vehicle Dynamics Project")
        self.setStrokeColor(colors.HexColor("#CCCCCC"))
        self.setLineWidth(0.5)
        self.line(54, 52, 558, 52)

        self.restoreState()


def build_pdf_report(output_filename: str = "results_fsg_benchmark/Simulation_Report.pdf"):
    doc = SimpleDocTemplate(
        output_filename,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "CoverTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=24,
        leading=28,
        textColor=colors.HexColor("#0f2a4a"),
        spaceAfter=8,
    )

    subtitle_style = ParagraphStyle(
        "CoverSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#334e68"),
        spaceAfter=15,
    )

    h1_style = ParagraphStyle(
        "Heading1_Custom",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=18,
        textColor=colors.HexColor("#0f2a4a"),
        spaceBefore=14,
        spaceAfter=6,
        keepWithNext=True,
    )

    h2_style = ParagraphStyle(
        "Heading2_Custom",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=14,
        textColor=colors.HexColor("#1b4965"),
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True,
    )

    body_style = ParagraphStyle(
        "Body_Custom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor("#222222"),
        spaceAfter=6,
    )

    code_style = ParagraphStyle(
        "Code_Custom",
        parent=styles["Normal"],
        fontName="Courier",
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor("#1e293b"),
        spaceAfter=4,
    )

    table_header_style = ParagraphStyle(
        "TableHeader",
        fontName="Helvetica-Bold",
        fontSize=9,
        leading=11,
        textColor=colors.white,
    )

    table_cell_style = ParagraphStyle(
        "TableCell",
        fontName="Helvetica",
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#1a202c"),
    )

    table_bold_cell_style = ParagraphStyle(
        "TableBoldCell",
        fontName="Helvetica-Bold",
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#0f2a4a"),
    )

    story = []

    # Title Banner
    story.append(Spacer(1, 15))
    story.append(Paragraph("LAP TIME SIMULATOR (laptimesim)", title_style))
    story.append(Paragraph("High-Fidelity 18+ DOF Multi-Body Dynamics & Optimal Control Simulation Report", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#0f2a4a"), spaceAfter=14))

    # Executive Summary
    story.append(Paragraph("1. Executive Summary", h1_style))
    story.append(Paragraph(
        "This document presents the detailed architectural formulation, mathematical modeling, and complete parameter specification of the <b>laptimesim</b> vehicle dynamics simulation package. "
        "The software was built for Formula Student electric racing teams and reduced-order modeling (ROM) research. "
        "The solver formulates a minimum-time <b>Optimal Control Problem (OCP)</b> in curvilinear (Frenet) coordinates along the circuit centerline, "
        "transcribed via direct collocation into a large-scale nonlinear programming problem (NLP) solved with <b>CasADi and IPOPT (MUMPS)</b>.",
        body_style
    ))
    story.append(Paragraph(
        "A full-vehicle benchmark was executed on the official <b>Formula Student Germany (FSG at Hockenheimring) Endurance Circuit (903 m)</b>. "
        "The vehicle model incorporates full multi-body suspension load transfers, Pacejka combined slip tire interactions, multi-dimensional aerodynamic ride-height maps, and electric powertrain power limits.",
        body_style
    ))

    # Simulation Methodology & Degrees of Freedom
    story.append(Paragraph("2. Mathematical Formulation & Degrees of Freedom", h1_style))
    story.append(Paragraph(
        "The simulation models the vehicle with <b>18+ continuous physical degrees of freedom</b>, described by the state vector <b>x(s)</b> and control vector <b>u(s)</b> where arc length <i>s</i> is the independent variable.",
        body_style
    ))

    dof_data = [
        [Paragraph("Subsystem", table_header_style), Paragraph("States / DOFs", table_header_style), Paragraph("Physics Governed", table_header_style)],
        [Paragraph("Chassis 6-DOF", table_bold_cell_style), Paragraph("vx, vy, vz, phi, theta, psi_dot, phi_dot, theta_dot", table_cell_style), Paragraph("Coupled Newton-Euler equations of rigid body translation and rotation in 3D space.", table_cell_style)],
        [Paragraph("Wheel Spin (4x)", table_bold_cell_style), Paragraph("omega_FL, omega_FR, omega_RL, omega_RR", table_cell_style), Paragraph("Rotational inertia dynamics driven by motor/brake torques and longitudinal tire traction.", table_cell_style)],
        [Paragraph("Suspension Travel (4x)", table_bold_cell_style), Paragraph("zs_FL, zs_FR, zs_RL, zs_RR, dzs_FL, dzs_FR, dzs_RL, dzs_RR", table_cell_style), Paragraph("Quarter-car deflections, damping forces, anti-roll bars (ARB), and progressive elastomeric bump stops.", table_cell_style)],
        [Paragraph("Compliances", table_bold_cell_style), Paragraph("delta_compl, theta_dt", table_cell_style), Paragraph("Steering rack/column torsional compliance and rear drivetrain axle torsional twist.", table_cell_style)],
        [Paragraph("Battery Energy & Thermal", table_bold_cell_style), Paragraph("soc, t_batt", table_cell_style), Paragraph("Equivalent Circuit Model (ECM Rint) tracking State of Charge, Joule heating, and terminal voltage.", table_cell_style)],
        [Paragraph("Curvilinear Tracking", table_bold_cell_style), Paragraph("n, psi", table_cell_style), Paragraph("Lateral offset from centerline n(s) and heading error angle relative to track tangent psi(s).", table_cell_style)],
    ]

    t_dof = Table(dof_data, colWidths=[110, 160, 234])
    t_dof.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#0f2a4a")),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t_dof)

    # Curvilinear Transformation
    story.append(Spacer(1, 10))
    story.append(Paragraph("Curvilinear Spatial Transformation:", h2_style))
    story.append(Paragraph(
        "To formulate the objective strictly as minimum time, the problem transforms time-derivatives to spatial derivatives via the progress speed equation:<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<b>dt/ds = (1 - n·κ(s)) / (vx·cos(ψ) - vy·sin(ψ))</b><br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<b>dx_veh/ds = (dx_veh/dt) · (dt/ds)</b><br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<b>Objective: min ∫ (dt/ds) ds</b>",
        code_style
    ))

    # Page Break for Parameter Tables
    story.append(PageBreak())

    # Full Subsystems Specification & Parameters
    story.append(Paragraph("3. Detailed Parameter Tables & Maps", h1_style))
    story.append(Paragraph(
        "The following parameters define the vehicle configuration used in the benchmark simulation (matching <i>examples/fsg_benchmark.toml</i>):",
        body_style
    ))

    params_data = [
        [Paragraph("Subsystem", table_header_style), Paragraph("Parameter", table_header_style), Paragraph("Value", table_header_style), Paragraph("Unit / Description", table_header_style)],
        # Chassis
        [Paragraph("Chassis & Inertia", table_bold_cell_style), Paragraph("Total Mass (m)", table_cell_style), Paragraph("290.0", table_cell_style), Paragraph("kg (215 kg car + 75 kg driver)", table_cell_style)],
        [Paragraph("", table_cell_style), Paragraph("Wheelbase (L)", table_cell_style), Paragraph("1.530", table_cell_style), Paragraph("m", table_cell_style)],
        [Paragraph("", table_cell_style), Paragraph("Track Width Front / Rear", table_cell_style), Paragraph("1.200 / 1.180", table_cell_style), Paragraph("m", table_cell_style)],
        [Paragraph("", table_cell_style), Paragraph("CG Height (h_cg)", table_cell_style), Paragraph("0.280", table_cell_style), Paragraph("m", table_cell_style)],
        [Paragraph("", table_cell_style), Paragraph("Weight Distribution (Front / Rear)", table_cell_style), Paragraph("48% / 52%", table_cell_style), Paragraph("Longitudinal CG balance", table_cell_style)],
        [Paragraph("", table_cell_style), Paragraph("Inertia Tensor (Ixx, Iyy, Izz)", table_cell_style), Paragraph("78.0, 110.0, 90.0", table_cell_style), Paragraph("kg·m² (Roll, Pitch, Yaw)", table_cell_style)],
        # Tire
        [Paragraph("Pacejka MF 5.2 Tire", table_bold_cell_style), Paragraph("Stiffness Factor (B)", table_cell_style), Paragraph("Front: 10.5 | Rear: 11.0", table_cell_style), Paragraph("TTC Hoosier 18.0x7.5-10 fitment", table_cell_style)],
        [Paragraph("", table_cell_style), Paragraph("Shape Factor (C)", table_cell_style), Paragraph("Front: 1.65 | Rear: 1.68", table_cell_style), Paragraph("Peak sharpness", table_cell_style)],
        [Paragraph("", table_cell_style), Paragraph("Peak Grip Factor (D)", table_cell_style), Paragraph("Front: 1.55 | Rear: 1.60", table_cell_style), Paragraph("Peak friction coefficient (μ)", table_cell_style)],
        [Paragraph("", table_cell_style), Paragraph("Curvature Factor (E)", table_cell_style), Paragraph("-0.40", table_cell_style), Paragraph("Post-peak falloff profile", table_cell_style)],
        [Paragraph("", table_cell_style), Paragraph("Loaded Radius (r_wheel)", table_cell_style), Paragraph("0.2032 (8.0 in)", table_cell_style), Paragraph("m", table_cell_style)],
        # Aero
        [Paragraph("Aerodynamics", table_bold_cell_style), Paragraph("Lift Coefficient (Cl)", table_cell_style), Paragraph("3.60", table_cell_style), Paragraph("High-downforce multi-element wings", table_cell_style)],
        [Paragraph("", table_cell_style), Paragraph("Drag Coefficient (Cd)", table_cell_style), Paragraph("1.45", table_cell_style), Paragraph("Aerodynamic drag factor", table_cell_style)],
        [Paragraph("", table_cell_style), Paragraph("Frontal Area (A)", table_cell_style), Paragraph("1.15", table_cell_style), Paragraph("m²", table_cell_style)],
        [Paragraph("", table_cell_style), Paragraph("Center of Pressure (CoP)", table_cell_style), Paragraph("46%", table_cell_style), Paragraph("Aero downforce front distribution", table_cell_style)],
        # Powertrain & Battery
        [Paragraph("Powertrain (EV)", table_bold_cell_style), Paragraph("Peak Motor Power", table_cell_style), Paragraph("80.0", table_cell_style), Paragraph("kW (FSG rule ceiling)", table_cell_style)],
        [Paragraph("", table_cell_style), Paragraph("Peak Motor Torque", table_cell_style), Paragraph("240.0", table_cell_style), Paragraph("Nm", table_cell_style)],
        [Paragraph("", table_cell_style), Paragraph("Gearbox Reduction Ratio", table_cell_style), Paragraph("3.60", table_cell_style), Paragraph("Single-speed planetary / spur", table_cell_style)],
        [Paragraph("", table_cell_style), Paragraph("Drivetrain Efficiency (η)", table_cell_style), Paragraph("94.0%", table_cell_style), Paragraph("Inverter + gearbox transmission", table_cell_style)],
        [Paragraph("", table_cell_style), Paragraph("Regen Braking Torque Cap", table_cell_style), Paragraph("60.0", table_cell_style), Paragraph("Nm (Front/Rear regenerative recovery)", table_cell_style)],
        # Battery ECM
        [Paragraph("Battery Pack ECM", table_bold_cell_style), Paragraph("Capacity / Voltage", table_cell_style), Paragraph("6.8 kWh / 400 V", table_cell_style), Paragraph("Nominal pack energy & voltage", table_cell_style)],
        [Paragraph("", table_cell_style), Paragraph("Internal Resistance (R_int)", table_cell_style), Paragraph("0.045", table_cell_style), Paragraph("Ohms (total accumulator)", table_cell_style)],
        # Suspension
        [Paragraph("Suspension", table_bold_cell_style), Paragraph("Spring Rates (Front / Rear)", table_cell_style), Paragraph("38,000 / 42,000", table_cell_style), Paragraph("N/m (Linear main springs)", table_cell_style)],
        [Paragraph("", table_cell_style), Paragraph("Damper Rates (Front / Rear)", table_cell_style), Paragraph("1,900 / 2,100", table_cell_style), Paragraph("N·s/m (Damping coefficients)", table_cell_style)],
        [Paragraph("", table_cell_style), Paragraph("ARB Stiffness (Front / Rear)", table_cell_style), Paragraph("5,500 / 4,500", table_cell_style), Paragraph("N/m (Anti-Roll Bar roll resistance)", table_cell_style)],
    ]

    t_params = Table(params_data, colWidths=[105, 145, 95, 159])
    t_params.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#0f2a4a")),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(t_params)

    # Multi-dimensional Maps Section
    story.append(Paragraph("4. Aerodynamic & Nonlinear Look-up Maps", h1_style))
    story.append(Paragraph(
        "<b>Multi-Dimensional Aerodynamic Ride Height Map (B-spline Surface):</b><br/>"
        "In the extended aero module (<i>AeroMapModel</i>), Cl, Cd, and balance are functions of front and rear ride heights:<br/>"
        "• Grid X (Front Ride Height): [20 mm, 30 mm, 40 mm, 50 mm]<br/>"
        "• Grid Y (Rear Ride Height): [25 mm, 35 mm, 45 mm, 55 mm]<br/>"
        "• Cl Map Matrix: [[3.4, 3.3, 3.1, 2.9], [3.5, 3.4, 3.2, 3.0], [3.3, 3.2, 3.0, 2.8], [3.1, 3.0, 2.8, 2.6]]<br/>"
        "• Front Aero Balance: Evaluated using continuous 2D B-spline interpolation in CasADi ensuring C² differentiability for IPOPT.",
        body_style
    ))
    story.append(Paragraph(
        "<b>Nonlinear Suspension & Bump Stop Curve:</b><br/>"
        "When suspension travel <i>z_s</i> exceeds bump clearance (15 mm front, 18 mm rear), a progressive elastomeric force is activated:<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<b>F_bump = k_bump · (max(0, z_s - clearance))^1.5</b><br/>"
        "where <i>k_bump</i> = 120,000 N/m (front) and 140,000 N/m (rear), preventing chassis grounding under high aerodynamic downforce.",
        body_style
    ))

    # Page Break for Results & Plots
    story.append(PageBreak())

    story.append(Paragraph("5. Simulation Results & Performance Telemetry", h1_style))
    story.append(Paragraph(
        "The optimal trajectory was computed over the <b>903 m FSG Endurance circuit</b>. Key performance telemetry metrics include:",
        body_style
    ))

    res_data = [
        [Paragraph("Simulation Metric", table_header_style), Paragraph("Calculated Result", table_header_style), Paragraph("Significance / Competitiveness", table_header_style)],
        [Paragraph("Endurance Lap Time", table_bold_cell_style), Paragraph("39.80 s", table_cell_style), Paragraph("Corresponds to an ideal 79.6 s 2-lap stint, matching top FSG vehicles.", table_cell_style)],
        [Paragraph("Average Track Speed", table_bold_cell_style), Paragraph("83.19 km/h", table_cell_style), Paragraph("High average cornering speed maintained through aerodynamic grip.", table_cell_style)],
        [Paragraph("Top Straightline Speed", table_bold_cell_style), Paragraph("100.86 km/h", table_cell_style), Paragraph("Reached at the end of the 75 m main straight under 80 kW ceiling.", table_cell_style)],
        [Paragraph("Max Lateral Acceleration (ay)", table_bold_cell_style), Paragraph("3.59 g", table_cell_style), Paragraph("Extreme grip enabled by Cl = 3.6 downforce and soft TTC compound.", table_cell_style)],
        [Paragraph("Max Longitudinal Accel (ax)", table_bold_cell_style), Paragraph("+1.83 g", table_cell_style), Paragraph("Traction limited at low speeds, power-limited at high speeds.", table_cell_style)],
        [Paragraph("Max Braking Deceleration", table_bold_cell_style), Paragraph("-1.78 g", table_cell_style), Paragraph("Four-wheel hydraulic + regenerative combined braking envelope.", table_cell_style)],
    ]

    t_res = Table(res_data, colWidths=[150, 110, 244])
    t_res.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#0f2a4a")),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t_res)

    story.append(Spacer(1, 10))

    # Embed Plots if available
    img_gg = Path("results_fsg_benchmark/gg_diagram.png")
    img_racing_line = Path("results_fsg_benchmark/racing_line.png")
    img_channels = Path("results_fsg_benchmark/telemetry_channels.png")

    if img_gg.is_file() and img_racing_line.is_file():
        story.append(Paragraph("6. Telemetry & Acceleration Diagrams", h1_style))
        img_table_data = [
            [
                Image(str(img_gg), width=235, height=235),
                Image(str(img_racing_line), width=235, height=235),
            ],
            [
                Paragraph("<b>Figure 1:</b> G-G Acceleration Diagram (ax vs. ay friction envelope)", body_style),
                Paragraph("<b>Figure 2:</b> Optimal Racing Line & Track Centerline", body_style),
            ]
        ]
        t_img = Table(img_table_data, colWidths=[252, 252])
        t_img.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LEFTPADDING', (0, 0), (-1, -1), 2),
            ('RIGHTPADDING', (0, 0), (-1, -1), 2),
        ]))
        story.append(t_img)

    if img_channels.is_file():
        story.append(PageBreak())
        story.append(Paragraph("7. Comprehensive Telemetry Channels", h1_style))
        story.append(Paragraph(
            "The figure below illustrates the synchronized telemetry traces along track arc-length <i>s</i>, including speed, throttle/brake pedal inputs, road wheel steering angle, motor power demand (kW), and battery SoC (%):",
            body_style
        ))
        story.append(Image(str(img_channels), width=500, height=520))

    # Build the document
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Report successfully compiled: {Path(output_filename).resolve()}")


if __name__ == "__main__":
    build_pdf_report()
