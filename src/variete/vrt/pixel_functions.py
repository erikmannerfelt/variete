from typing import Literal
from variete import misc
import xml.etree.ElementTree as ET


class PixelFunction:
    name: str
    arguments: dict[str, str] | None
    code: str | None
    language: Literal["Python"] | None

    def __init__(
        self,
        name: str,
        arguments: dict[str, str] | None = None,
        code: str | None = None,
        language: Literal["Python"] | None = None,
    ):
        for attr in ["arguments", "name", "code", "language"]:
            setattr(self, attr, locals()[attr])

    def __repr__(self):
        code_string = f" {len(self.code)} character custom code" if self.code is not None else ""
        return "\n".join([f"PixelFunction: {self.name}({self.arguments or ''}){code_string}"])

    def validate(self, n_bands: int):
        assert len(self.name) > 0, "The PixelFunction must have a name"
        assert self.language is None or self.language == "Python", "PixelFunction language must be None or 'Python'"

        assert n_bands > 0, "PixelFunction requires at least one band"

        if self.code is not None and self.language is None:
            raise AssertionError("PixelFunction code is given but the language is not provided")

    def to_etree_keys(self) -> list[ET.Element]:

        keys = []

        keys.append(misc.new_element("PixelFunctionType", self.name))
        keys.append(misc.new_element("PixelFunctionArguments", None, self.arguments))

        if self.language is not None:
            keys.append(misc.new_element("PixelFunctionLanguage", self.language))

        if self.code is not None:
            keys.append(misc.new_element("PixelFunctionCode", self.code))

        return keys


class SumPixelFunction(PixelFunction):
    def __init__(self, constant: int | float | None = None):
        self.name = "sum"
        self.arguments = {"k": misc.number_to_gdal(constant)} if constant is not None else None
        self.code = self.language = None

    def validate(self, n_bands: int):
        PixelFunction.validate(self, n_bands=n_bands)

        if "k" in (self.arguments or {}):
            n_bands += 1

        if n_bands < 2:
            raise ValueError("SumPixelFunction requires at least two bands (or one band and a constant")


class MulPixelFunction(PixelFunction):
    def __init__(self, constant: int | float | None = None):
        base = SumPixelFunction(constant=constant)
        self.name = "mul"
        for attr in ["arguments", "code", "language"]:
            setattr(self, attr, getattr(base, attr))


class DivPixelFunction(PixelFunction):
    def __init__(self, constant: int | float | None = None):
        base = SumPixelFunction(constant=constant)
        self.name = "div"
        for attr in ["arguments", "code", "language"]:
            setattr(self, attr, getattr(base, attr))


class InvPixelFunction(PixelFunction):
    def __init__(self):
        self.name = "inv"
        self.arguments = self.code = self.language = None


class ScalePixelFunction(PixelFunction):
    def __init__(self):
        self.name = "scale"
        self.arguments = self.code = self.language = None


AnyPixelFunction = (
    PixelFunction | SumPixelFunction | ScalePixelFunction | MulPixelFunction | DivPixelFunction | InvPixelFunction
)


def pixel_function_from_etree(elem: ET.Element) -> AnyPixelFunction:

    if (pixel_function_type_elem := elem.find("PixelFunctionType")) is not None:
        name = pixel_function_type_elem.text
    else:
        raise ValueError("Key PixelFunctionType does not exist. Invalid PixelFunction")

    language = getattr(elem.find("PixelFunctionLanguage"), "text", None)

    if (arguments_elem := elem.find("PixelFunctionArguments")) is not None:
        arguments = dict(arguments_elem.items())
    else:
        arguments = None

    code = getattr(elem.find("PixelFunctionCode"), "text", None)

    if code is None:
        if name == "sum":
            return SumPixelFunction(arguments.get("k", None))
        if name == "scale":
            return ScalePixelFunction()
        if name == "mul":
            return MulPixelFunction(arguments.get("k", None))
        if name == "div":
            return DivPixelFunction(arguments.get("k", None))
        if name == "inv":
            return InvPixelFunction()
        raise ValueError(f"Empty PixelFunctionCode and unknown type: {name}")

    return PixelFunction(name=name, arguments=arguments, code=code, language=language)
