"""List packaged scenes that reference the platformer controller; never launch Unity."""
import argparse
from pathlib import Path

import UnityPy


def inspect(data):
    settings = UnityPy.load(str(data / "globalgamemanagers"))
    scenes = next(o.read().scenes for o in settings.objects if o.type.name == "BuildSettings")
    assets = UnityPy.load(str(data / "globalgamemanagers.assets"))
    scripts = {o.path_id: o.read().m_ClassName for o in assets.objects if o.type.name == "MonoScript"}
    for index, scene in enumerate(scenes):
        level = UnityPy.load(str(data / f"level{index}"))
        attached = set()
        for obj in level.objects:
            if obj.type.name != "MonoBehaviour":
                continue
            ref = obj.parse_monobehaviour_head().m_Script
            if ref.m_FileID and obj.assets_file.externals[ref.m_FileID - 1].path == "globalgamemanagers.assets":
                attached.add(scripts[ref.m_PathID])
        if "RickController" in attached:
            yield index, scene, "LevelEndTrigger" in attached, "GameSessionManager" in attached


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path, help="Installed NeonSky_Data directory")
    print("index\tscene\tend_trigger\tsession_manager")
    for row in inspect(parser.parse_args().data):
        print(*row, sep="\t")
