import os
import platform
from manim import *

# ============================================================
# LPV-ARX Regression Visualization
# Clean educational animation for YouTube
# ============================================================

class RegressionExplanation(Scene):
    def construct(self):

        # ----------------------------------------------------
        # Title
        # ----------------------------------------------------

        title = Text(
            "Building the Regression Problem",
            font_size=42
        ).to_edge(UP*1.5)

        self.play(Write(title))
        self.wait(0.7)

        # ----------------------------------------------------
        # Step 1 — Build feature rows
        # ----------------------------------------------------

        sample_title = Text(
            "Each row in X represents one timestep",
            font_size=30
        ).next_to(title, DOWN, buff=0.3)

        self.play(FadeIn(sample_title, shift=UP))

        # Example row contents
        row = MathTex(
            r"x[k] = ",
            r"[",
            r"a_y[k-1],\, \dots,\, a_y[k-n_{ay}]",
            r",",
            r"\delta[k],\, \dots,\, \delta[k-n_{\delta}]",
            r",",
            r"\mathrm{roll}[k]",
            r"]",
            font_size=40
        )

        row.set_color_by_tex("a_y", BLUE)
        row.set_color_by_tex(r"\delta", GREEN)
        row.set_color_by_tex(r"\mathrm{roll}", ORANGE)

        row.move_to(UP * 0.5)

        self.play(Write(row))
        self.wait(1.0)

        # ----------------------------------------------------
        # Step 2 — LPV expansion
        # ----------------------------------------------------

        lpv_text = Text(
            "Each regressor is expanded by LPV basis functions",
            font_size=28
        ).next_to(row, DOWN, buff=0.9)

        self.play(FadeIn(lpv_text, shift=UP))

        basis = MathTex(
            r"\phi(v) = [1,\ v,\ v^2]",
            font_size=42
        )

        basis.set_color(YELLOW)
        basis.next_to(lpv_text, DOWN, buff=0.4)

        self.play(Write(basis))
        self.wait(0.8)

        expanded = MathTex(
            r"\;\rightarrow\;",
            r"x[k]\cdot \phi(v[k])",
            font_size=42
        )

        expanded.next_to(basis, DOWN, buff=0.4)

        self.play(Write(expanded[0]))
        self.play(Write(expanded[1]))

        self.wait(1.5)

        # ----------------------------------------------------
        # Step 3 — Build X matrix
        # ----------------------------------------------------

        self.play(
            FadeOut(expanded),
            FadeOut(row),
            FadeOut(lpv_text),
            FadeOut(basis),
        )

        matrix_title = Text(
            "All samples are stacked into a large feature matrix",
            font_size=30
        ).next_to(title, DOWN, buff=0.3)

        self.play(Transform(sample_title, matrix_title))

        X_matrix = MathTex(
            r"X =",
            r"\begin{bmatrix}"
            r"x[k_1] \cdot \phi(v[k_1]) \\"
            r"x[k_2] \cdot \phi(v[k_2]) \\"
            r"\vdots \\"
            r"x[k_N] \cdot \phi(v[k_N])"
            r"\end{bmatrix}",
            font_size=42
        )

        X_matrix.move_to(LEFT * 3)

        self.play(Write(X_matrix))
        self.wait(1)

        # ----------------------------------------------------
        # Theta vector
        # ----------------------------------------------------

        theta = MathTex(
            r"\theta =",
            r"\begin{bmatrix}"
            r"\theta_1 \\"
            r"\theta_2 \\"
            r"\theta_3 \\"
            r"\vdots"
            r"\end{bmatrix}",
            font_size=42
        )

        theta.move_to(RIGHT * 1.5)

        theta_text = Text(
            "Unknown coefficients",
            font_size=20, slant=ITALIC
        ).next_to(theta, DOWN)

        self.play(Write(theta))
        self.play(FadeIn(theta_text))

        self.wait(1)

        # ----------------------------------------------------
        # Multiplication
        # ----------------------------------------------------

        equation = MathTex(
            r"X\theta",
            r"\approx",
            r"y",
            font_size=52
        )

        equation.move_to(DOWN * 2.7)

        equation[0].set_color(BLUE)
        equation[2].set_color(YELLOW)

        y_text = Text(
            "Measured next lateral acceleration",
            font_size=20, slant=ITALIC
        ).next_to(equation[2], DOWN)

        self.play(Write(equation))
        self.play(FadeIn(y_text))

        self.wait(1.5)

        # ----------------------------------------------------
        # Least squares objective
        # ----------------------------------------------------

        self.play(
            FadeOut(X_matrix),
            FadeOut(theta),
            FadeOut(theta_text),
            FadeOut(y_text),
            FadeOut(sample_title),
        )

        objective = MathTex(
            r"\min_{\theta}",
            r"\|X\theta - y\|^2",
            font_size=58
        )

        objective[1].set_color(BLUE)

        self.play(
            Transform(equation, objective)
        )

        self.wait(1)

        subtitle = Text(
            "Find parameters that minimize prediction error",
            font_size=30
        ).next_to(objective, DOWN, buff=0.7)

        self.play(FadeIn(subtitle))

        self.wait(2)

        # ----------------------------------------------------
        # Final result
        # ----------------------------------------------------

        final_text = Text(
            "Solve for θ using least squares",
            font_size=38
        )

        final_text.set_color(GREEN)

        final_text.move_to(DOWN * 2.7)

        self.play(Write(final_text))

        self.wait(3)



if __name__ == "__main__":
    from manim import config

    # Optional: Set output quality here (low or high)
    config.quality = "low_quality"    # Change to "high_quality" for higher resolution
    config.media_dir = os.getcwd()    # Optional: Set output directory

    # Render the scene
    RegressionExplanation().render()

    # Automatically open the output file
    if platform.system() == 'Windows':
        os.system(f"start {config.output_file}")
        input("Press Enter to exit...")  # Prevents the script from closing immediately. Need in Windows for some reason
    else:
        os.system(f"xdg-open {config.output_file}")        
