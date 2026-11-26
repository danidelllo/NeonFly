using System;
using System.Collections.Generic;
using System.Reflection;
using System.Reflection.Emit;
using HarmonyLib;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace NeonFly
{
    public sealed class OwnedProjectile : MonoBehaviour { }
    public sealed class DestroyedCombatTarget : MonoBehaviour { }

    internal static class Combat
    {
        internal static readonly CombatLedger Ledger = new CombatLedger();

        public static GameObject SpawnOwned(GameObject prefab, Vector3 position, Quaternion rotation)
        {
            var result = UnityEngine.Object.Instantiate(prefab, position, rotation);
            if (!result.GetComponent<OwnedProjectile>()) result.AddComponent<OwnedProjectile>();
            return result;
        }

        public static void Damage(EnemyHealth target, int amount, MonoBehaviour source)
        {
            // Capture identity before TakeDamage can destroy the target.
            int before = target.CurrentHealth, maximum = target.MaxHealth;
            int id = target.GetInstanceID(), scene = SceneManager.GetActiveScene().handle;
            bool owned = source && source.GetComponentInParent<OwnedProjectile>() &&
                         !target.GetComponent<DestroyedCombatTarget>();
            target.TakeDamage(amount); // Preserve the actual game implementation.
            Ledger.Record(scene, id, before, target.CurrentHealth, maximum, owned);
        }
    }

    // Some collision/explosion paths destroy a target without reducing its
    // health first. Prevent a projectile later in that frame from earning
    // credit against that already-destroying target.
    [HarmonyPatch]
    internal static class MarkDestroyedTargets
    {
        private static IEnumerable<MethodBase> TargetMethods()
        {
            yield return AccessTools.Method(typeof(EnemyHealth), "Die");
            yield return AccessTools.Method(typeof(EnemyGroundA), "Explode");
            yield return AccessTools.Method(typeof(EnemyJetA), "Explode");
            yield return AccessTools.Method(typeof(EnemySoldierA), "Die");
            yield return AccessTools.Method(typeof(EnemySoldierB), "Die");
        }

        private static void Prefix(MonoBehaviour __instance)
        {
            var target = __instance.GetComponent<EnemyHealth>();
            if (target && !target.GetComponent<DestroyedCombatTarget>())
                target.gameObject.AddComponent<DestroyedCombatTarget>();
        }
    }

    [HarmonyPatch]
    internal static class MarkPlayerProjectiles
    {
        private static IEnumerable<MethodBase> TargetMethods()
        {
            yield return AccessTools.Method(typeof(Player), "Shoot");
            yield return AccessTools.Method(typeof(Player), "ShootRocket");
        }

        private static IEnumerable<CodeInstruction> Transpiler(IEnumerable<CodeInstruction> instructions)
        {
            int changed = 0;
            foreach (var ins in instructions) {
                var method = ins.operand as MethodInfo;
                if (ins.opcode == OpCodes.Call && method != null && method.DeclaringType == typeof(UnityEngine.Object) &&
                    method.Name == "Instantiate" && method.IsGenericMethod &&
                    method.GetGenericArguments()[0] == typeof(GameObject) && method.GetParameters().Length == 3 &&
                    method.GetParameters()[1].ParameterType == typeof(Vector3) &&
                    method.GetParameters()[2].ParameterType == typeof(Quaternion)) {
                    ins.operand = AccessTools.Method(typeof(Combat), "SpawnOwned"); changed++;
                }
                yield return ins;
            }
            if (changed != 1) throw new InvalidOperationException("Unrecognized player projectile spawn signature");
        }
    }

    [HarmonyPatch]
    internal static class CreditPlayerWeaponDamage
    {
        private static IEnumerable<MethodBase> TargetMethods()
        {
            yield return AccessTools.Method(typeof(PlayerBullet), "OnTriggerEnter2D");
            yield return AccessTools.Method(typeof(Rocket), "OnTriggerEnter2D");
            yield return AccessTools.Method(typeof(Rocket), "ApplyAOEDamage");
            yield return AccessTools.Method(typeof(Rocket), "ApplyRadialSpreadDamage");
            yield return AccessTools.Method(typeof(ElectromagneticWave), "OnTriggerEnter2D");
            // This is an enemy turret rocket in the installed game. Wrapping
            // it exercises the same path, but its unmarked damage earns zero.
            yield return AccessTools.Method(typeof(HomingRocket), "Explode");
        }

        private static IEnumerable<CodeInstruction> Transpiler(IEnumerable<CodeInstruction> instructions)
        {
            var original = AccessTools.Method(typeof(EnemyHealth), "TakeDamage", new[] { typeof(int) });
            int changed = 0;
            foreach (var ins in instructions) {
                if ((ins.opcode == OpCodes.Callvirt || ins.opcode == OpCodes.Call) && Equals(ins.operand, original)) {
                    var source = new CodeInstruction(OpCodes.Ldarg_0);
                    source.labels.AddRange(ins.labels); ins.labels.Clear();
                    yield return source;
                    ins.opcode = OpCodes.Call;
                    ins.operand = AccessTools.Method(typeof(Combat), "Damage"); changed++;
                }
                yield return ins;
            }
            if (changed == 0) throw new InvalidOperationException("Unrecognized weapon damage signature");
        }
    }
}
