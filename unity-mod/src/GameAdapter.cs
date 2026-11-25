using System;
using System.Collections.Generic;
using System.Reflection;
using System.Reflection.Emit;
using HarmonyLib;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.UI;
using UObject = UnityEngine.Object;

namespace NeonFly
{
    internal static class GameAdapter
    {
        internal static int Remaining, Applied, Direction, Deaths, Completions;
        internal static int ActiveHandle;
        internal static double DeathSeconds = -1, CompletionSeconds = -1;
        internal static double DamageFraction;
        internal static int VerticalDirection, Weapons;
        internal static bool FullInput;
        internal static int LastTickFrame = -1;
        internal static Player Ship;
        internal static bool Driving
        {
            get { return Remaining > 0 && Ship && SceneManager.GetActiveScene().handle == ActiveHandle; }
        }

        internal static void Step(int direction, int ticks, bool full)
        {
            Ship = UObject.FindObjectOfType<Player>();
            if (!Ship || !Ship.isActiveAndEnabled || UObject.FindObjectOfType<RickController>() ||
                Time.timeScale <= 0 || ticks < 1 || ticks > 32 ||
                (full ? direction < 0 || direction >= 36 : Math.Abs(direction) > 1))
                throw new InvalidOperationException("ship_step_unavailable");
            ActiveHandle = SceneManager.GetActiveScene().handle;
            Applied = 0;
            FullInput = full;
            Direction = full ? Axis(direction % 3) : direction;
            VerticalDirection = full ? Axis(direction / 3 % 3) : 0;
            Weapons = full ? direction / 9 : 0;
            LastTickFrame = -1;
            Remaining = ticks;
        }

        private static int Axis(int digit) { return digit == 1 ? -1 : digit == 2 ? 1 : 0; }

        internal static bool Held(KeyCode physical, Player actor, int mask)
        {
            // Unity runs Update after FixedUpdate; include the render frame of the last tick.
            if (FullInput && actor == Ship && SceneManager.GetActiveScene().handle == ActiveHandle &&
                (Remaining > 0 || Time.frameCount == LastTickFrame))
                return (Weapons & mask) != 0;
            return Input.GetKey(physical);
        }

        internal static string[] Bindings()
        {
            var c = SettingsSystem.Instance ? SettingsSystem.Instance.CurrentSettings : new SettingsData();
            return new[] { "shipLeft="+c.shipLeft, "shipRight="+c.shipRight, "shipDown="+c.shipDown,
                "shipUp="+c.shipUp, "shipShoot="+c.shipShoot, "shipRocket="+c.shipRocket,
                "rickLeft="+c.rickLeft, "rickRight="+c.rickRight, "rickJump="+c.rickJump,
                "rickCrouch="+c.rickCrouch, "rickShoot="+c.rickShoot, "rickGrenade="+c.rickGrenade };
        }

        internal static void Reset()
        {
            var manager = UObject.FindObjectOfType<GameSessionManager>();
            if (!manager || !SceneManager.GetActiveScene().name.StartsWith("Level_", StringComparison.Ordinal))
                throw new InvalidOperationException("reset_unavailable");
            manager.Play();
        }

        internal static string[] Buttons()
        {
            var names = new List<string>();
            foreach (var b in UObject.FindObjectsOfType<Button>())
                if (b.isActiveAndEnabled && b.IsInteractable())
                    for (int i = 0; i < b.onClick.GetPersistentEventCount(); i++)
                        names.Add(b.onClick.GetPersistentMethodName(i));
            return names.ToArray();
        }

        internal static void Click(int op)
        {
            string scene = SceneManager.GetActiveScene().name;
            string method = op == 3 ? "NewGame" : "LoadLevel1";
            if ((op == 3 && (scene != "MainMenu" || SaveManager.SaveFileExists())) ||
                (op == 4 && scene != "LevelSelect"))
                throw new InvalidOperationException("menu_guard");
            foreach (var b in UObject.FindObjectsOfType<Button>())
                if (b.isActiveAndEnabled && b.IsInteractable())
                    for (int i = 0; i < b.onClick.GetPersistentEventCount(); i++)
                        if (b.onClick.GetPersistentMethodName(i) == method)
                        {
                            b.onClick.Invoke();
                            return;
                        }
            throw new InvalidOperationException("button_unavailable");
        }
    }

    [HarmonyPatch(typeof(Player), "GetHorizontalInput")]
    internal static class Horizontal
    {
        private static void Postfix(Player __instance, ref float __result)
        {
            if (GameAdapter.Driving && __instance == GameAdapter.Ship) __result = GameAdapter.Direction;
        }
    }

    [HarmonyPatch(typeof(Player), "FixedUpdate")]
    internal static class Tick
    {
        private static void Postfix(Player __instance)
        {
            if (GameAdapter.Driving && __instance == GameAdapter.Ship)
            {
                GameAdapter.Remaining--;
                GameAdapter.Applied++;
                GameAdapter.LastTickFrame = Time.frameCount;
            }
        }
    }

    [HarmonyPatch(typeof(Player), "GetVerticalInput")]
    internal static class Vertical
    {
        private static void Postfix(Player __instance, ref float __result)
        {
            if (GameAdapter.FullInput && GameAdapter.Driving && __instance == GameAdapter.Ship)
                __result = GameAdapter.VerticalDirection;
        }
    }

    [HarmonyPatch(typeof(Player), "Update")]
    internal static class WeaponKeys
    {
        public static bool Shoot(KeyCode key, Player actor) { return GameAdapter.Held(key, actor, 1); }
        public static bool Rocket(KeyCode key, Player actor) { return GameAdapter.Held(key, actor, 2); }

        private static IEnumerable<CodeInstruction> Transpiler(IEnumerable<CodeInstruction> instructions)
        {
            var getKey = AccessTools.Method(typeof(Input), "GetKey", new[] {typeof(KeyCode)});
            FieldInfo field = null;
            int gun = 0, rocket = 0;
            foreach (var ins in instructions)
            {
                if (ins.opcode == OpCodes.Call && Equals(ins.operand, getKey))
                {
                    if (field == null || field.DeclaringType != typeof(SettingsData) ||
                        (field.Name != "shipShoot" && field.Name != "shipRocket"))
                        throw new InvalidOperationException("Unrecognized Player.Update key read");
                    bool primary = field.Name == "shipShoot";
                    if (primary) gun++; else rocket++;
                    ins.opcode = OpCodes.Ldarg_0;
                    ins.operand = null;
                    yield return ins;
                    yield return new CodeInstruction(OpCodes.Call,
                        AccessTools.Method(typeof(WeaponKeys), primary ? "Shoot" : "Rocket"));
                }
                else yield return ins;
                field = ins.opcode == OpCodes.Ldfld ? ins.operand as FieldInfo : null;
            }
            if (gun != 1 || rocket != 1) throw new InvalidOperationException("Player.Update input signature changed");
        }
    }

    [HarmonyPatch(typeof(PlayerHealth), "Die")]
    internal static class Death
    {
        private static void Postfix(PlayerHealth __instance)
        {
            if (__instance.GetComponent<Player>())
            {
                GameAdapter.Deaths++;
                GameAdapter.DeathSeconds = Time.fixedTimeAsDouble;
            }
        }
    }

    [HarmonyPatch(typeof(PlayerHealth), "TakeDamage")]
    internal static class Damage
    {
        private static void Prefix(PlayerHealth __instance, out int __state) { __state = __instance.CurrentHealth; }
        private static void Postfix(PlayerHealth __instance, int __state) {
            if (__instance.GetComponent<Player>() && __instance.MaxHealth > 0)
                GameAdapter.DamageFraction += Math.Max(0, __state - Math.Max(0, __instance.CurrentHealth)) / (double)__instance.MaxHealth;
        }
    }

    [HarmonyPatch(typeof(LevelEndTrigger), "OnTriggerEnter2D")]
    internal static class Finish
    {
        private static void Prefix(bool ___triggered, out bool __state) { __state = ___triggered; }
        private static void Postfix(bool ___triggered, bool __state)
        {
            if (!__state && ___triggered)
            {
                GameAdapter.Completions++;
                GameAdapter.CompletionSeconds = Time.fixedTimeAsDouble;
            }
        }
    }
}
