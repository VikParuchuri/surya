from mathml_to_latex import MathMLToLaTeX


def mathml_to_latex(mathml: str) -> str:
    try:
        return MathMLToLaTeX().convert(mathml)
    except Exception as e:
        return mathml
