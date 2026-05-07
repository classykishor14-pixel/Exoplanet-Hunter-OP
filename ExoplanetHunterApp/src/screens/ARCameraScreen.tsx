import React from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  Animated,
} from 'react-native';
import { Colors, FontFamily, Spacing, BorderRadius } from '../theme';

export default function ARCameraScreen() {
  const scanAnim = React.useRef(new Animated.Value(0)).current;
  const radarAnim = React.useRef(new Animated.Value(0)).current;

  React.useEffect(() => {
    Animated.loop(
      Animated.timing(scanAnim, {
        toValue: 1,
        duration: 2500,
        useNativeDriver: true,
      }),
    ).start();

    Animated.loop(
      Animated.sequence([
        Animated.timing(radarAnim, {
          toValue: 1,
          duration: 1500,
          useNativeDriver: true,
        }),
        Animated.timing(radarAnim, {
          toValue: 0,
          duration: 1500,
          useNativeDriver: true,
        }),
      ]),
    ).start();
  }, [scanAnim, radarAnim]);

  const scanY = scanAnim.interpolate({
    inputRange: [0, 1],
    outputRange: [0, 260],
  });

  return (
    <View style={styles.container}>
      <Text style={styles.heading}>AR STAR MAP</Text>
      <Text style={styles.subheading}>Point your camera at the night sky</Text>

      {/* Camera viewfinder mockup */}
      <View style={styles.viewfinder}>
        {/* Corner brackets */}
        <View style={[styles.corner, styles.cornerTL]} />
        <View style={[styles.corner, styles.cornerTR]} />
        <View style={[styles.corner, styles.cornerBL]} />
        <View style={[styles.corner, styles.cornerBR]} />

        {/* Scan line */}
        <Animated.View
          style={[styles.scanLine, { transform: [{ translateY: scanY }] }]}
        />

        {/* Radar rings */}
        <Animated.View style={[styles.radarRing, styles.radarRing1, { opacity: radarAnim }]} />
        <Animated.View style={[styles.radarRing, styles.radarRing2, { opacity: radarAnim, transform: [{ scale: radarAnim.interpolate({ inputRange: [0, 1], outputRange: [0.5, 1.2] }) }] }]} />

        {/* Star annotations */}
        {[
          { top: '20%', left: '30%', name: 'Kepler-452' },
          { top: '50%', left: '65%', name: 'TRAPPIST-1' },
          { top: '70%', left: '25%', name: 'Proxima Cen' },
        ].map((star) => (
          <View key={star.name} style={[styles.starAnnotation, { top: star.top as any, left: star.left as any }]}>
            <View style={styles.starDot} />
            <View style={styles.starLine} />
            <Text style={styles.starLabel}>{star.name}</Text>
          </View>
        ))}

        {/* Center crosshair */}
        <View style={styles.crosshair}>
          <View style={styles.crosshairH} />
          <View style={styles.crosshairV} />
        </View>

        {/* Status badge */}
        <Animated.View style={[styles.statusBadge, { opacity: radarAnim }]}>
          <Text style={styles.statusText}>● SCANNING</Text>
        </Animated.View>
      </View>

      {/* Bottom controls */}
      <View style={styles.controls}>
        <TouchableOpacity style={styles.controlBtn}>
          <Text style={styles.controlIcon}>⭐</Text>
          <Text style={styles.controlLabel}>Stars</Text>
        </TouchableOpacity>
        <TouchableOpacity style={[styles.controlBtn, styles.captureBtn]}>
          <Text style={styles.captureText}>CAPTURE</Text>
        </TouchableOpacity>
        <TouchableOpacity style={styles.controlBtn}>
          <Text style={styles.controlIcon}>🪐</Text>
          <Text style={styles.controlLabel}>Planets</Text>
        </TouchableOpacity>
      </View>

      <Text style={styles.disclaimer}>
        AR mode requires camera permission and clear night sky
      </Text>
    </View>
  );
}

const CORNER_SIZE = 24;

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: Colors.background,
    paddingTop: 60,
    paddingHorizontal: Spacing.lg,
    alignItems: 'center',
  },
  heading: {
    fontFamily: FontFamily.bold,
    fontSize: 26,
    color: Colors.accent,
    letterSpacing: 6,
    textShadowColor: Colors.accent,
    textShadowOffset: { width: 0, height: 0 },
    textShadowRadius: 10,
  },
  subheading: {
    fontFamily: FontFamily.regular,
    fontSize: 11,
    color: Colors.textMuted,
    letterSpacing: 1,
    marginTop: 4,
    marginBottom: Spacing.lg,
  },
  viewfinder: {
    width: '100%',
    height: 300,
    backgroundColor: Colors.surface,
    borderRadius: BorderRadius.lg,
    borderWidth: 1,
    borderColor: Colors.border,
    overflow: 'hidden',
    position: 'relative',
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: Spacing.xl,
  },
  corner: {
    position: 'absolute',
    width: CORNER_SIZE,
    height: CORNER_SIZE,
    borderColor: Colors.accent,
  },
  cornerTL: { top: 12, left: 12, borderTopWidth: 3, borderLeftWidth: 3, borderTopLeftRadius: 4 },
  cornerTR: { top: 12, right: 12, borderTopWidth: 3, borderRightWidth: 3, borderTopRightRadius: 4 },
  cornerBL: { bottom: 12, left: 12, borderBottomWidth: 3, borderLeftWidth: 3, borderBottomLeftRadius: 4 },
  cornerBR: { bottom: 12, right: 12, borderBottomWidth: 3, borderRightWidth: 3, borderBottomRightRadius: 4 },
  scanLine: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    height: 2,
    backgroundColor: Colors.accent,
    opacity: 0.7,
    shadowColor: Colors.accent,
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 1,
    shadowRadius: 8,
    elevation: 8,
  },
  radarRing: {
    position: 'absolute',
    width: 120,
    height: 120,
    borderRadius: 60,
    borderWidth: 1,
    borderColor: Colors.accent,
  },
  radarRing1: { opacity: 0.3 },
  radarRing2: { width: 180, height: 180, borderRadius: 90 },
  starAnnotation: {
    position: 'absolute',
    alignItems: 'flex-start',
    flexDirection: 'row',
  },
  starDot: {
    width: 6,
    height: 6,
    borderRadius: 3,
    backgroundColor: Colors.accent,
    shadowColor: Colors.accent,
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 1,
    shadowRadius: 4,
    elevation: 4,
    marginTop: 2,
  },
  starLine: {
    width: 20,
    height: 1,
    backgroundColor: Colors.accent,
    opacity: 0.5,
    marginTop: 5,
  },
  starLabel: {
    fontFamily: FontFamily.regular,
    fontSize: 9,
    color: Colors.accent,
    marginLeft: 2,
    marginTop: -2,
  },
  crosshair: {
    position: 'absolute',
    width: 40,
    height: 40,
    alignItems: 'center',
    justifyContent: 'center',
  },
  crosshairH: { position: 'absolute', width: 40, height: 1, backgroundColor: Colors.accent, opacity: 0.4 },
  crosshairV: { position: 'absolute', width: 1, height: 40, backgroundColor: Colors.accent, opacity: 0.4 },
  statusBadge: {
    position: 'absolute',
    bottom: 12,
    right: 12,
    backgroundColor: Colors.accentDim,
    borderRadius: BorderRadius.full,
    paddingHorizontal: Spacing.sm,
    paddingVertical: 3,
  },
  statusText: {
    fontFamily: FontFamily.regular,
    fontSize: 9,
    color: Colors.accent,
    letterSpacing: 1,
  },
  controls: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: Spacing.lg,
    marginBottom: Spacing.md,
  },
  controlBtn: {
    alignItems: 'center',
    gap: 4,
  },
  controlIcon: { fontSize: 28 },
  controlLabel: {
    fontFamily: FontFamily.regular,
    fontSize: 10,
    color: Colors.textMuted,
    letterSpacing: 1,
  },
  captureBtn: {
    backgroundColor: Colors.accentDim,
    borderWidth: 2,
    borderColor: Colors.accent,
    borderRadius: BorderRadius.full,
    width: 80,
    height: 80,
    alignItems: 'center',
    justifyContent: 'center',
    shadowColor: Colors.accent,
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.8,
    shadowRadius: 16,
    elevation: 16,
  },
  captureText: {
    fontFamily: FontFamily.bold,
    fontSize: 10,
    color: Colors.accent,
    letterSpacing: 2,
  },
  disclaimer: {
    fontFamily: FontFamily.regular,
    fontSize: 10,
    color: Colors.textMuted,
    textAlign: 'center',
    marginTop: Spacing.sm,
  },
});
