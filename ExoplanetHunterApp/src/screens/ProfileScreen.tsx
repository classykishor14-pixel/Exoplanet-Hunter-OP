import React from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  ScrollView,
  Animated,
} from 'react-native';
import { Colors, FontFamily, Spacing, BorderRadius } from '../theme';

const ACHIEVEMENTS = [
  { icon: '🌍', label: 'First Contact', desc: 'Viewed first exoplanet' },
  { icon: '🔭', label: 'Deep Observer', desc: 'Searched 50 planets' },
  { icon: '📡', label: 'Signal Hunter', desc: 'Used AR map 10 times' },
  { icon: '🚀', label: 'Pioneer', desc: 'Joined the mission' },
];

const STATS = [
  { label: 'Planets Viewed', value: '142' },
  { label: 'AR Sessions', value: '28' },
  { label: 'Bookmarks', value: '17' },
  { label: 'Days Active', value: '63' },
];

export default function ProfileScreen() {
  const ringAnim = React.useRef(new Animated.Value(0)).current;

  React.useEffect(() => {
    Animated.loop(
      Animated.sequence([
        Animated.timing(ringAnim, { toValue: 1, duration: 3000, useNativeDriver: true }),
        Animated.timing(ringAnim, { toValue: 0, duration: 3000, useNativeDriver: true }),
      ]),
    ).start();
  }, [ringAnim]);

  const ringOpacity = ringAnim.interpolate({ inputRange: [0, 1], outputRange: [0.3, 0.8] });

  return (
    <ScrollView
      style={styles.container}
      contentContainerStyle={styles.contentContainer}
      showsVerticalScrollIndicator={false}
    >
      {/* Avatar area */}
      <View style={styles.avatarSection}>
        <Animated.View style={[styles.avatarRing, { opacity: ringOpacity }]} />
        <View style={styles.avatar}>
          <Text style={styles.avatarEmoji}>👨‍🚀</Text>
        </View>
        <Text style={styles.username}>COMMANDER</Text>
        <Text style={styles.handle}>@exohunter_001</Text>
        <View style={styles.levelBadge}>
          <Text style={styles.levelText}>⭐ LEVEL 12 · STELLAR CARTOGRAPHER</Text>
        </View>
      </View>

      {/* Stats grid */}
      <Text style={styles.sectionTitle}>MISSION STATS</Text>
      <View style={styles.statsGrid}>
        {STATS.map((s) => (
          <View key={s.label} style={styles.statCell}>
            <Text style={styles.statValue}>{s.value}</Text>
            <Text style={styles.statLabel}>{s.label}</Text>
          </View>
        ))}
      </View>

      {/* Achievements */}
      <Text style={styles.sectionTitle}>ACHIEVEMENTS</Text>
      {ACHIEVEMENTS.map((a) => (
        <View key={a.label} style={styles.achievementRow}>
          <View style={styles.achievementIcon}>
            <Text style={{ fontSize: 22 }}>{a.icon}</Text>
          </View>
          <View style={{ flex: 1 }}>
            <Text style={styles.achievementLabel}>{a.label}</Text>
            <Text style={styles.achievementDesc}>{a.desc}</Text>
          </View>
          <View style={styles.achievementCheck}>
            <Text style={styles.checkmark}>✓</Text>
          </View>
        </View>
      ))}

      {/* Settings */}
      <Text style={styles.sectionTitle}>SETTINGS</Text>
      {['Notification Preferences', 'Data Sources', 'AR Calibration', 'About'].map((item) => (
        <TouchableOpacity key={item} style={styles.menuItem}>
          <Text style={styles.menuLabel}>{item}</Text>
          <Text style={styles.menuArrow}>›</Text>
        </TouchableOpacity>
      ))}

      <View style={{ height: 100 }} />
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: Colors.background },
  contentContainer: { paddingTop: 60, paddingHorizontal: Spacing.lg },
  avatarSection: { alignItems: 'center', marginBottom: Spacing.xl },
  avatarRing: {
    position: 'absolute',
    width: 110,
    height: 110,
    borderRadius: 55,
    borderWidth: 2,
    borderColor: Colors.accent,
    top: -5,
    shadowColor: Colors.accent,
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.8,
    shadowRadius: 12,
    elevation: 12,
  },
  avatar: {
    width: 90,
    height: 90,
    borderRadius: 45,
    backgroundColor: Colors.surfaceLight,
    borderWidth: 2,
    borderColor: Colors.border,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: Spacing.md,
  },
  avatarEmoji: { fontSize: 44 },
  username: {
    fontFamily: FontFamily.bold,
    fontSize: 22,
    color: Colors.text,
    letterSpacing: 6,
  },
  handle: {
    fontFamily: FontFamily.regular,
    fontSize: 12,
    color: Colors.textMuted,
    marginTop: 4,
    marginBottom: Spacing.md,
  },
  levelBadge: {
    backgroundColor: Colors.accentDim,
    borderWidth: 1,
    borderColor: Colors.accent,
    borderRadius: BorderRadius.full,
    paddingVertical: 4,
    paddingHorizontal: Spacing.md,
  },
  levelText: {
    fontFamily: FontFamily.regular,
    fontSize: 9,
    color: Colors.accent,
    letterSpacing: 1,
  },
  sectionTitle: {
    fontFamily: FontFamily.bold,
    fontSize: 12,
    color: Colors.accent,
    letterSpacing: 4,
    marginBottom: Spacing.md,
    marginTop: Spacing.lg,
  },
  statsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: Spacing.sm,
    marginBottom: Spacing.sm,
  },
  statCell: {
    flex: 1,
    minWidth: '45%',
    backgroundColor: Colors.surface,
    borderWidth: 1,
    borderColor: Colors.border,
    borderRadius: BorderRadius.md,
    padding: Spacing.md,
    alignItems: 'center',
  },
  statValue: {
    fontFamily: FontFamily.bold,
    fontSize: 24,
    color: Colors.accent,
  },
  statLabel: {
    fontFamily: FontFamily.regular,
    fontSize: 10,
    color: Colors.textMuted,
    marginTop: 2,
    letterSpacing: 0.5,
  },
  achievementRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: Spacing.md,
    backgroundColor: Colors.surface,
    borderWidth: 1,
    borderColor: Colors.border,
    borderRadius: BorderRadius.md,
    padding: Spacing.md,
    marginBottom: Spacing.sm,
  },
  achievementIcon: {
    width: 44,
    height: 44,
    borderRadius: 22,
    backgroundColor: Colors.surfaceLight,
    alignItems: 'center',
    justifyContent: 'center',
  },
  achievementLabel: {
    fontFamily: FontFamily.bold,
    fontSize: 13,
    color: Colors.text,
  },
  achievementDesc: {
    fontFamily: FontFamily.regular,
    fontSize: 10,
    color: Colors.textMuted,
    marginTop: 2,
  },
  achievementCheck: {
    width: 24,
    height: 24,
    borderRadius: 12,
    backgroundColor: Colors.accentDim,
    borderWidth: 1,
    borderColor: Colors.accent,
    alignItems: 'center',
    justifyContent: 'center',
  },
  checkmark: { color: Colors.accent, fontSize: 12, fontFamily: FontFamily.bold },
  menuItem: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    backgroundColor: Colors.surface,
    borderWidth: 1,
    borderColor: Colors.border,
    borderRadius: BorderRadius.md,
    padding: Spacing.md,
    marginBottom: Spacing.sm,
  },
  menuLabel: {
    fontFamily: FontFamily.regular,
    fontSize: 14,
    color: Colors.text,
  },
  menuArrow: {
    fontFamily: FontFamily.bold,
    fontSize: 20,
    color: Colors.textMuted,
  },
});
