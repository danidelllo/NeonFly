using System;
using UnityEngine;

namespace NeonFly
{
    [Serializable]
    public sealed class RangeScan
    {
        public bool valid;
        public bool insideView;
        public float range = 12;
        public float[] distance = new float[32];
        public float[] visible = new float[32];
        public float[] visibleStart = new float[32];
        public int[] kind = new int[32];
        public float[] geometryDistance = new float[32];
        public int[] geometryKind = new int[32];
        public float vx, vy, screenX, screenY;
    }

    internal static class RangeSensor
    {
        // Slab intersection of a world-distance ray with the camera viewport.
        // Keeping both entry and exit allows off-screen origins without querying
        // off-screen geometry. Distances remain measured from the actual ship.
        internal static bool ClipViewport(float x, float y, float dx, float dy, float range,
                                          out float enter, out float exit)
        {
            enter = 0; exit = range;
            return ClipAxis(x, dx, ref enter, ref exit) &&
                   ClipAxis(y, dy, ref enter, ref exit) && exit - enter > 1e-5f;
        }

        private static bool ClipAxis(float origin, float direction, ref float enter, ref float exit)
        {
            if (Math.Abs(direction) < 1e-7f) return origin >= 0 && origin <= 1;
            float a = -origin / direction, b = (1 - origin) / direction;
            enter = Math.Max(enter, Math.Min(a, b));
            exit = Math.Min(exit, Math.Max(a, b));
            return exit >= enter;
        }

        // Categories: 0 clear/unknown outside view; 1 solid; 2 damageable;
        // 3 hostile projectile; 4 lethal. No level-specific names or coordinates.
        private static int Category(Collider2D collider, Player player)
        {
            if (collider.GetComponentInParent<Player>() == player) return -1;
            foreach (var component in collider.GetComponentsInParent<MonoBehaviour>()) {
                if (!component) continue;
                string name = component.GetType().Name;
                bool projectile = name.Contains("Bullet") || name.Contains("Rocket") ||
                                  name.Contains("Missile") || name.Contains("Grenade");
                if (projectile && (name.StartsWith("Player", StringComparison.Ordinal) ||
                                   name.StartsWith("Rick", StringComparison.Ordinal))) return -1;
                if (name == "EnemyBullet" || name == "EnemyAirRocket" || name == "EnemyHomingAirMissile" ||
                    name == "EnemyGrenade" || name == "EnemySoldierBullet" || name == "SentryGunBullet") return 3;
            }
            if (collider.GetComponentInParent<DeathZone>()) return 4;
            if (collider.GetComponentInParent<EnemyHealth>()) return 2;
            return collider.isTrigger ? -1 : 1;
        }

        internal static RangeScan Capture(Player player)
        {
            var result = new RangeScan();
            var camera = Camera.main;
            if (!player || !player.isActiveAndEnabled || !camera || !camera.orthographic) return result;
            var collider = player.GetComponentInChildren<Collider2D>();
            Vector3 origin = collider ? collider.bounds.center : player.transform.position;
            Vector3 viewport = camera.WorldToViewportPoint(origin);
            if (viewport.z <= 0) return result;
            var body = player.GetComponent<Rigidbody2D>();
            result.vx = body ? body.velocity.x : 0;
            result.vy = body ? body.velocity.y : 0;
            result.screenX = viewport.x; result.screenY = viewport.y;
            result.valid = true;
            result.insideView = viewport.x >= 0 && viewport.x <= 1 && viewport.y >= 0 && viewport.y <= 1;
            var filter = new ContactFilter2D();
            filter.SetLayerMask(Physics2D.AllLayers);
            filter.useTriggers = true;
            var hits = new RaycastHit2D[1024];
            for (int i = 0; i < 32; i++) {
                float angle = i * 2 * Mathf.PI / 32;
                var direction = new Vector2(Mathf.Cos(angle), Mathf.Sin(angle));
                Vector3 delta = camera.WorldToViewportPoint(origin + (Vector3)direction) - viewport;
                float enter, limit;
                if (!ClipViewport(viewport.x, viewport.y, delta.x, delta.y, result.range,
                                  out enter, out limit)) continue;
                result.visibleStart[i] = enter;
                result.visible[i] = limit;
                result.distance[i] = limit;
                result.geometryDistance[i] = limit;
                // Query only the visible segment, even when the ship is off screen.
                int count = Physics2D.Raycast((Vector2)origin + direction * enter, direction,
                                              filter, hits, limit - enter);
                if (count == hits.Length) throw new InvalidOperationException("sensor_hit_limit");
                for (int j = 0; j < count; j++) {
                    var hit = hits[j];
                    if (!hit.collider) continue;
                    int category = Category(hit.collider, player);
                    if (category < 0) continue;
                    float distance = enter + hit.distance;
                    if (category != 3 && distance <= result.geometryDistance[i]) {
                        result.geometryDistance[i] = distance; result.geometryKind[i] = category;
                    }
                    if (distance <= result.distance[i]) {
                        result.distance[i] = distance; result.kind[i] = category;
                    }
                }
            }
            result.valid = true;
            return result;
        }
    }
}
