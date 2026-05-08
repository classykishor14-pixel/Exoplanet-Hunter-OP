import React, { useEffect, useRef } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  Animated,
  Dimensions,
  TouchableOpacity,
} from 'react-native';
import { useRoute, useNavigation } from '@react-navigation/native';
import Svg, { Circle } from 'react-native-svg';
import { Colors, FontFamily, Spacing, BorderRadius } from '../theme';
import { SearchResult, Analysis } from '../services/api';

const { width } = Dimensions.get('window');

interface PlanetDetailsRouteParams {
  result: SearchResult;
}

interface HabitabilityRingProps {
  habitabilityScore: number;
  isHabitable: boolean;
}

const HabitabilityRing: React.FC<HabitabilityRingProps> = ({ habitabilityScore, isHabitable }) => {
  const animatedValue = useRef(new Animated.Value(0)).current;
  const scaleAnim = useRef(new Animated.Value(0.8)).current;

  useEffect(() => {
    Animated.parallel([
      Animated.timing(animatedValue, {
        toValue: 1,
        duration: 1500,
        useNativeDriver: true,
      }),
      Animated.spring(scaleAnim, {
        toValue: 1,
        useNativeDriver: true,
      }),
    ]).start();
  }, []);

  const circumference = 2 * Math.PI * 45;
  const strokeDashoffset = (1 - habitabilityScore / 100) * circumference;

  return (
    <View style={styles.habitabilityRing}>
      <Animated.View style={{ transform: [{ scale: scaleAnim }] }}>
        <Svg height={120} width={120} viewBox="0 0 120 120">
          {/* Background circle */}
          <Circle
            cx="60"
            cy="60"
            r="45"
            stroke={Colors.border}
            strokeWidth="2"
            fill="none"
          />
          {/* Progress circle - animated via strokeDashoffset */}
          <Circle
            cx="60"
            cy="60"
            r="45"
            stroke={isHabitable ? Colors.success : Colors.accent}
            strokeWidth="3"
            fill="none"
            strokeDasharray={circumference}
            strokeDashoffset={strokeDashoffset}
            strokeLinecap="round"
          />
        </Svg>
      </Animated.View>
      <View style={styles.habitabilityCenter}>
        <Animated.Text
          style={[
            styles.habitabilityScore,
            {
              color: isHabitable ? Colors.success : Colors.accent,
              opacity: animatedValue,
            },
          ]}
        >
          {habitabilityScore.toFixed(0)}%
        </Animated.Text>
        <Text style={styles.habitabilityLabel}>{isHabitable ? 'Habitable' : 'Not Ideal'}</Text>
      </View>
    </View>
  );
};

interface InfoCardProps {
  title: string;
  icon: string;
  children: React.ReactNode;
  delay: number;
}

const InfoCard: React.FC<InfoCardProps> = ({ title, icon, children, delay }) => {
  const fadeAnim = useRef(new Animated.Value(0)).current;
  const scaleAnim = useRef(new Animated.Value(0.8)).current;

  useEffect(() => {
    Animated.parallel([
      Animated.timing(fadeAnim, {
        toValue: 1,
        duration: 600,
        delay,
        useNativeDriver: true,
      }),
      Animated.timing(scaleAnim, {
        toValue: 1,
        duration: 600,
        delay,
        useNativeDriver: true,
      }),
    ]).start();
  }, []);

  return (
    <Animated.View
      style={[
        styles.infoCard,
        { opacity: fadeAnim, transform: [{ scale: scaleAnim }] },
      ]}
    >
      <View style={styles.cardHeader}>
        <Text style={styles.cardIcon}>{icon}</Text>
        <Text style={styles.cardTitle}>{title}</Text>
      </View>
      <View style={styles.cardContent}>{children}</View>
    </Animated.View>
  );
};

export default function PlanetDetailsScreen() {
  const route = useRoute<any>();
  const navigation = useNavigation();
  const { result } = route.params as PlanetDetailsRouteParams;

  const headerFadeAnim = useRef(new Animated.Value(0)).current;
  const headerScaleAnim = useRef(new Animated.Value(0.8)).current;

  useEffect(() => {
    Animated.parallel([
      Animated.timing(headerFadeAnim, {
        toValue: 1,
        duration: 500,
        useNativeDriver: true,
      }),
      Animated.timing(headerScaleAnim, {
        toValue: 1,
        duration: 500,
        useNativeDriver: true,
      }),
    ]).start();
  }, []);

  const { planet, analysis, raw_data } = result;
  const habitabilityScore = analysis.is_in_habitable_zone ? 85 : 45;
  const compositionEmoji = getCompositionEmoji(analysis.planet_type);
  const atmosphereRetention = getAtmosphereRetention(analysis.atmosphere_prediction);

  return (
    <View style={styles.container}>
      <ScrollView showsVerticalScrollIndicator={false} contentContainerStyle={styles.scrollContent}>
        {/* Header Section */}
        <Animated.View
          style={[
            styles.header,
            { opacity: headerFadeAnim, transform: [{ scale: headerScaleAnim }] },
          ]}
        >
          <Text style={styles.planetName}>{planet}</Text>
          <View style={styles.badgeContainer}>
            <View style={styles.compositionBadge}>
              <Text style={styles.badgeEmoji}>{compositionEmoji}</Text>
              <Text style={styles.badgeText}>{analysis.planet_type}</Text>
            </View>
          </View>
        </Animated.View>

        {/* Habitability Ring */}
        <InfoCard title="Habitability Score" icon="🎯" delay={100}>
          <HabitabilityRing
            habitabilityScore={habitabilityScore}
            isHabitable={analysis.is_in_habitable_zone}
          />
        </InfoCard>

        {/* Info Grid */}
        <View style={styles.gridContainer}>
          {/* Composition Card */}
          <View style={styles.gridRow}>
            <View style={styles.gridCol}>
              <InfoCard title="Composition" icon={compositionEmoji} delay={200}>
                <Text style={styles.infoText}>{analysis.composition_estimate}</Text>
              </InfoCard>
            </View>

            {/* Atmosphere Card */}
            <View style={styles.gridCol}>
              <InfoCard title="Atmosphere" icon="🌬️" delay={300}>
                <View style={styles.atmosphereContainer}>
                  <View style={styles.retentionBar}>
                    <View
                      style={[
                        styles.retentionFill,
                        { width: `${atmosphereRetention}%` },
                      ]}
                    />
                  </View>
                  <Text style={styles.atmosphereText}>{atmosphereRetention}% Retention</Text>
                  <Text style={styles.predictionText}>{analysis.atmosphere_prediction}</Text>
                </View>
              </InfoCard>
            </View>
          </View>

          {/* Habitable Zone Card */}
          <View style={styles.gridRow}>
            <View style={styles.gridCol}>
              <InfoCard title="Habitable Zone" icon="🌍" delay={400}>
                <View style={styles.habitableZoneContent}>
                  <Text style={styles.zoneLabel}>
                    {analysis.is_in_habitable_zone ? '✓ IN ZONE' : '✗ OUT OF ZONE'}
                  </Text>
                  {analysis.habitable_zone_inner_au && analysis.habitable_zone_outer_au && (
                    <Text style={styles.zoneRange}>
                      {analysis.habitable_zone_inner_au.toFixed(3)} - {analysis.habitable_zone_outer_au.toFixed(3)} AU
                    </Text>
                  )}
                </View>
              </InfoCard>
            </View>

            {/* Quick Stats Card */}
            <View style={styles.gridCol}>
              <InfoCard title="Quick Stats" icon="📊" delay={500}>
                <View style={styles.statsContainer}>
                  {raw_data.pl_orbsmax && (
                    <View style={styles.statRow}>
                      <Text style={styles.statLabel}>SMA:</Text>
                      <Text style={styles.statValue}>{raw_data.pl_orbsmax.toFixed(3)} AU</Text>
                    </View>
                  )}
                  {raw_data.pl_rade && (
                    <View style={styles.statRow}>
                      <Text style={styles.statLabel}>Radius:</Text>
                      <Text style={styles.statValue}>{raw_data.pl_rade.toFixed(2)} R⊕</Text>
                    </View>
                  )}
                  {raw_data.pl_masse && (
                    <View style={styles.statRow}>
                      <Text style={styles.statLabel}>Mass:</Text>
                      <Text style={styles.statValue}>{raw_data.pl_masse.toFixed(2)} M⊕</Text>
                    </View>
                  )}
                  {analysis.density_g_cm3 && (
                    <View style={styles.statRow}>
                      <Text style={styles.statLabel}>Density:</Text>
                      <Text style={styles.statValue}>{analysis.density_g_cm3.toFixed(2)} g/cm³</Text>
                    </View>
                  )}
                </View>
              </InfoCard>
            </View>
          </View>
        </View>

        <View style={{ height: 40 }} />
      </ScrollView>

      {/* Back Button */}
      <TouchableOpacity
        style={styles.backButton}
        onPress={() => navigation.goBack()}
      >
        <Text style={styles.backButtonText}>← Back</Text>
      </TouchableOpacity>
    </View>
  );
}

function getCompositionEmoji(planetType: string): string {
  const type = planetType.toLowerCase();
  if (type.includes('gas')) return '🪐';
  if (type.includes('rocky') || type.includes('earth')) return '🪨';
  if (type.includes('lava') || type.includes('hot')) return '🌋';
  return '🌍';
}

function getAtmosphereRetention(prediction: string): number {
  const pred = prediction.toLowerCase();
  if (pred.includes('thick')) return 85;
  if (pred.includes('moderate')) return 60;
  if (pred.includes('thin')) return 30;
  if (pred.includes('none') || pred.includes('little')) return 5;
  return 50;
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: Colors.background,
  },
  scrollContent: {
    paddingHorizontal: Spacing.lg,
    paddingTop: Spacing.lg,
    paddingBottom: Spacing.xxl,
  },
  header: {
    alignItems: 'center',
    marginBottom: Spacing.xxl,
  },
  planetName: {
    fontFamily: FontFamily.bold,
    fontSize: 36,
    color: Colors.accent,
    letterSpacing: 2,
    textShadowColor: Colors.accent,
    textShadowOffset: { width: 0, height: 0 },
    textShadowRadius: 15,
    marginBottom: Spacing.md,
  },
  badgeContainer: {
    flexDirection: 'row',
    gap: Spacing.sm,
  },
  compositionBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: Spacing.sm,
    backgroundColor: 'rgba(0, 255, 255, 0.1)',
    borderWidth: 1,
    borderColor: Colors.accent,
    borderRadius: BorderRadius.full,
    paddingHorizontal: Spacing.md,
    paddingVertical: Spacing.sm,
    shadowColor: Colors.accent,
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.4,
    shadowRadius: 8,
    elevation: 8,
  },
  badgeEmoji: {
    fontSize: 20,
  },
  badgeText: {
    fontFamily: FontFamily.bold,
    fontSize: 12,
    color: Colors.accent,
    letterSpacing: 1,
  },
  habitabilityRing: {
    alignItems: 'center',
    justifyContent: 'center',
    height: 140,
    marginVertical: Spacing.md,
  },
  habitabilityCenter: {
    position: 'absolute',
    alignItems: 'center',
    justifyContent: 'center',
  },
  habitabilityScore: {
    fontFamily: FontFamily.bold,
    fontSize: 24,
  },
  habitabilityLabel: {
    fontFamily: FontFamily.regular,
    fontSize: 10,
    color: Colors.textMuted,
    marginTop: 2,
  },
  infoCard: {
    backgroundColor: 'rgba(13, 21, 48, 0.8)',
    borderWidth: 1,
    borderColor: Colors.border,
    borderRadius: BorderRadius.md,
    padding: Spacing.md,
    backdropFilter: 'blur(10px)',
    shadowColor: Colors.accent,
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.15,
    shadowRadius: 6,
    elevation: 6,
  },
  cardHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: Spacing.sm,
    marginBottom: Spacing.sm,
  },
  cardIcon: {
    fontSize: 20,
  },
  cardTitle: {
    fontFamily: FontFamily.bold,
    fontSize: 12,
    color: Colors.accent,
    letterSpacing: 1,
  },
  cardContent: {
    gap: Spacing.sm,
  },
  infoText: {
    fontFamily: FontFamily.regular,
    fontSize: 13,
    color: Colors.text,
    lineHeight: 18,
  },
  gridContainer: {
    marginBottom: Spacing.lg,
  },
  gridRow: {
    flexDirection: 'row',
    gap: Spacing.md,
    marginBottom: Spacing.md,
  },
  gridCol: {
    flex: 1,
  },
  atmosphereContainer: {
    gap: Spacing.sm,
  },
  retentionBar: {
    height: 24,
    backgroundColor: 'rgba(0, 0, 0, 0.3)',
    borderRadius: BorderRadius.sm,
    overflow: 'hidden',
    borderWidth: 1,
    borderColor: Colors.border,
  },
  retentionFill: {
    height: '100%',
    backgroundColor: Colors.accent,
    borderRadius: BorderRadius.sm,
  },
  atmosphereText: {
    fontFamily: FontFamily.bold,
    fontSize: 11,
    color: Colors.accent,
  },
  predictionText: {
    fontFamily: FontFamily.regular,
    fontSize: 11,
    color: Colors.textMuted,
  },
  habitableZoneContent: {
    alignItems: 'center',
    gap: Spacing.sm,
  },
  zoneLabel: {
    fontFamily: FontFamily.bold,
    fontSize: 14,
    color: Colors.success,
    letterSpacing: 1,
  },
  zoneRange: {
    fontFamily: FontFamily.regular,
    fontSize: 11,
    color: Colors.textMuted,
  },
  statsContainer: {
    gap: Spacing.sm,
  },
  statRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  statLabel: {
    fontFamily: FontFamily.bold,
    fontSize: 11,
    color: Colors.textMuted,
  },
  statValue: {
    fontFamily: FontFamily.bold,
    fontSize: 12,
    color: Colors.accent,
  },
  backButton: {
    position: 'absolute',
    top: Spacing.lg,
    left: Spacing.lg,
    backgroundColor: 'rgba(13, 21, 48, 0.9)',
    borderWidth: 1,
    borderColor: Colors.border,
    borderRadius: BorderRadius.md,
    paddingHorizontal: Spacing.md,
    paddingVertical: Spacing.sm,
  },
  backButtonText: {
    fontFamily: FontFamily.regular,
    fontSize: 12,
    color: Colors.text,
  },
});
