import React, { useRef, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  Animated,
  TouchableOpacity,
  Dimensions,
} from 'react-native';
import { Colors, FontFamily, Spacing, BorderRadius } from '../theme';
import Starfield from '../components/Starfield';
import { useNavigation } from '@react-navigation/native';

const { width } = Dimensions.get('window');

export default function HomeScreen() {
  const navigation = useNavigation<any>();
  const glowAnim = useRef(new Animated.Value(0)).current;
  const fadeAnim = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    // Button glow loop
    Animated.loop(
      Animated.sequence([
        Animated.timing(glowAnim, {
          toValue: 1,
          duration: 1500,
          useNativeDriver: false, // Box shadow cannot use native driver
        }),
        Animated.timing(glowAnim, {
          toValue: 0,
          duration: 1500,
          useNativeDriver: false,
        }),
      ])
    ).start();

    // Initial fade in for content
    Animated.timing(fadeAnim, {
      toValue: 1,
      duration: 1000,
      delay: 500,
      useNativeDriver: true,
    }).start();
  }, [glowAnim, fadeAnim]);

  const buttonGlow = glowAnim.interpolate({
    inputRange: [0, 1],
    outputRange: [0, 15],
  });

  const buttonBorder = glowAnim.interpolate({
    inputRange: [0, 1],
    outputRange: [Colors.accentDim, Colors.accent],
  });

  return (
    <View style={styles.container}>
      {/* 3D Animated Starfield Background */}
      <Starfield />

      {/* Foreground Content */}
      <Animated.View style={[styles.content, { opacity: fadeAnim }]} pointerEvents="box-none">
        <View style={styles.heroSection}>
          {/* Logo / Title */}
          <Text style={styles.title}>EXOPLANET</Text>
          <Text style={styles.subtitle}>H U N T E R</Text>
          <View style={styles.accentLine} />
          
          {/* Tagline */}
          <Text style={styles.tagline}>Discover Worlds Beyond</Text>
        </View>

        {/* Action Section */}
        <View style={styles.actionSection}>
          <Animated.View
            style={[
              styles.buttonWrapper,
              {
                shadowRadius: buttonGlow,
                borderColor: buttonBorder,
              },
            ]}
          >
            <TouchableOpacity
              style={styles.button}
              activeOpacity={0.8}
              onPress={() => navigation.navigate('Search')}
            >
              <Text style={styles.buttonText}>START EXPLORING</Text>
            </TouchableOpacity>
          </Animated.View>
        </View>
      </Animated.View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: Colors.background,
  },
  content: {
    ...StyleSheet.absoluteFillObject,
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingVertical: 100,
    paddingHorizontal: Spacing.lg,
  },
  heroSection: {
    alignItems: 'center',
    marginTop: 40,
  },
  title: {
    fontFamily: FontFamily.bold,
    fontSize: 42,
    color: Colors.accent,
    letterSpacing: 8,
    textShadowColor: Colors.accent,
    textShadowOffset: { width: 0, height: 0 },
    textShadowRadius: 20,
    textAlign: 'center',
  },
  subtitle: {
    fontFamily: FontFamily.regular,
    fontSize: 24,
    color: Colors.white,
    letterSpacing: 16,
    marginTop: -8,
    textAlign: 'center',
  },
  accentLine: {
    width: 100,
    height: 3,
    backgroundColor: Colors.accent,
    borderRadius: 2,
    marginVertical: Spacing.lg,
    shadowColor: Colors.accent,
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 1,
    shadowRadius: 10,
    elevation: 10,
  },
  tagline: {
    fontFamily: FontFamily.regular,
    fontSize: 16,
    color: Colors.accent,
    letterSpacing: 4,
    textAlign: 'center',
    textShadowColor: Colors.accentDim,
    textShadowOffset: { width: 0, height: 0 },
    textShadowRadius: 8,
    textTransform: 'uppercase',
  },
  actionSection: {
    width: '100%',
    alignItems: 'center',
    marginBottom: 40,
  },
  buttonWrapper: {
    borderWidth: 2,
    borderRadius: BorderRadius.full,
    backgroundColor: 'rgba(0, 255, 255, 0.05)',
    shadowColor: Colors.accent,
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.8,
    elevation: 8,
    overflow: 'hidden',
  },
  button: {
    paddingVertical: Spacing.md,
    paddingHorizontal: Spacing.xxl,
    alignItems: 'center',
    justifyContent: 'center',
  },
  buttonText: {
    fontFamily: FontFamily.bold,
    fontSize: 16,
    color: Colors.accent,
    letterSpacing: 3,
    textShadowColor: Colors.accent,
    textShadowOffset: { width: 0, height: 0 },
    textShadowRadius: 5,
  },
});
