import React from 'react';
import {
  View,
  Text,
  StyleSheet,
  StatusBar,
  Animated,
  Dimensions,
} from 'react-native';
import { Colors, FontFamily, Spacing } from '../theme';

const { width } = Dimensions.get('window');

export default function HomeScreen() {
  const pulseAnim = React.useRef(new Animated.Value(0.4)).current;
  const glowAnim = React.useRef(new Animated.Value(0)).current;

  React.useEffect(() => {
    Animated.loop(
      Animated.sequence([
        Animated.timing(pulseAnim, {
          toValue: 1,
          duration: 2000,
          useNativeDriver: true,
        }),
        Animated.timing(pulseAnim, {
          toValue: 0.4,
          duration: 2000,
          useNativeDriver: true,
        }),
      ]),
    ).start();

    Animated.loop(
      Animated.sequence([
        Animated.timing(glowAnim, {
          toValue: 1,
          duration: 3000,
          useNativeDriver: true,
        }),
        Animated.timing(glowAnim, {
          toValue: 0,
          duration: 3000,
          useNativeDriver: true,
        }),
      ]),
    ).start();
  }, [pulseAnim, glowAnim]);

  const ringScale = glowAnim.interpolate({
    inputRange: [0, 1],
    outputRange: [0.9, 1.05],
  });

  return (
    <View style={styles.container}>
      <StatusBar barStyle="light-content" backgroundColor={Colors.background} />

      {/* Stars background dots */}
      <View style={styles.starsContainer} pointerEvents="none">
        {Array.from({ length: 40 }).map((_, i) => (
          <Animated.View
            key={i}
            style={[
              styles.star,
              {
                top: `${Math.random() * 100}%` as any,
                left: `${Math.random() * 100}%` as any,
                width: Math.random() > 0.7 ? 3 : 2,
                height: Math.random() > 0.7 ? 3 : 2,
                opacity: pulseAnim,
              },
            ]}
          />
        ))}
      </View>

      <View style={styles.content}>
        {/* Animated planet ring */}
        <Animated.View
          style={[styles.planetRing, { transform: [{ scale: ringScale }] }]}
        />
        <Animated.View
          style={[styles.planet, { opacity: pulseAnim }]}
        />

        {/* Title */}
        <Text style={styles.title}>EXOPLANET</Text>
        <Text style={styles.subtitle}>H U N T E R</Text>
        <View style={styles.accentLine} />
        <Text style={styles.tagline}>Explore worlds beyond our solar system</Text>

        {/* Stats row */}
        <View style={styles.statsRow}>
          {[
            { label: 'Confirmed', value: '5,600+' },
            { label: 'Candidates', value: '9,900+' },
            { label: 'Stars', value: '200B+' },
          ].map((stat) => (
            <View key={stat.label} style={styles.statCard}>
              <Text style={styles.statValue}>{stat.value}</Text>
              <Text style={styles.statLabel}>{stat.label}</Text>
            </View>
          ))}
        </View>

        {/* Mission badge */}
        <View style={styles.badge}>
          <Animated.View style={[styles.badgeDot, { opacity: pulseAnim }]} />
          <Text style={styles.badgeText}>LIVE MISSION DATA · NASA EXOPLANET ARCHIVE</Text>
        </View>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: Colors.background,
  },
  starsContainer: {
    ...StyleSheet.absoluteFillObject,
    overflow: 'hidden',
  },
  star: {
    position: 'absolute',
    backgroundColor: Colors.text,
    borderRadius: 99,
  },
  content: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: Spacing.lg,
    gap: Spacing.md,
  },
  planetRing: {
    width: 180,
    height: 180,
    borderRadius: 90,
    borderWidth: 2,
    borderColor: Colors.accent,
    position: 'absolute',
    top: '15%',
    opacity: 0.3,
  },
  planet: {
    width: 140,
    height: 140,
    borderRadius: 70,
    backgroundColor: Colors.accentDim,
    borderWidth: 2,
    borderColor: Colors.accent,
    marginBottom: Spacing.xl,
    shadowColor: Colors.accent,
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.8,
    shadowRadius: 24,
    elevation: 20,
  },
  title: {
    fontFamily: FontFamily.bold,
    fontSize: 36,
    color: Colors.accent,
    letterSpacing: 10,
    textShadowColor: Colors.accent,
    textShadowOffset: { width: 0, height: 0 },
    textShadowRadius: 16,
  },
  subtitle: {
    fontFamily: FontFamily.regular,
    fontSize: 22,
    color: Colors.text,
    letterSpacing: 14,
    marginTop: -8,
  },
  accentLine: {
    width: 80,
    height: 2,
    backgroundColor: Colors.accent,
    borderRadius: 2,
    marginVertical: Spacing.sm,
    shadowColor: Colors.accent,
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 1,
    shadowRadius: 8,
    elevation: 8,
  },
  tagline: {
    fontFamily: FontFamily.regular,
    fontSize: 12,
    color: Colors.textMuted,
    letterSpacing: 1,
    textAlign: 'center',
    marginBottom: Spacing.lg,
  },
  statsRow: {
    flexDirection: 'row',
    gap: Spacing.sm,
    marginVertical: Spacing.md,
  },
  statCard: {
    backgroundColor: Colors.surface,
    borderWidth: 1,
    borderColor: Colors.border,
    borderRadius: 12,
    paddingVertical: Spacing.sm,
    paddingHorizontal: Spacing.md,
    alignItems: 'center',
    minWidth: (width - Spacing.lg * 2 - Spacing.sm * 2) / 3,
  },
  statValue: {
    fontFamily: FontFamily.bold,
    fontSize: 16,
    color: Colors.accent,
  },
  statLabel: {
    fontFamily: FontFamily.regular,
    fontSize: 9,
    color: Colors.textMuted,
    letterSpacing: 1,
    marginTop: 2,
    textTransform: 'uppercase',
  },
  badge: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: Spacing.xs,
    backgroundColor: Colors.surfaceLight,
    borderWidth: 1,
    borderColor: Colors.border,
    borderRadius: 99,
    paddingVertical: 6,
    paddingHorizontal: Spacing.md,
    marginTop: Spacing.md,
  },
  badgeDot: {
    width: 6,
    height: 6,
    borderRadius: 3,
    backgroundColor: Colors.success,
  },
  badgeText: {
    fontFamily: FontFamily.regular,
    fontSize: 9,
    color: Colors.textMuted,
    letterSpacing: 1,
  },
});
