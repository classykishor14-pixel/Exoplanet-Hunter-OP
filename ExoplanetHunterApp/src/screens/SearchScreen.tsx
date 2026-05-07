import React from 'react';
import {
  View,
  Text,
  StyleSheet,
  TextInput,
  ScrollView,
  TouchableOpacity,
  Animated,
} from 'react-native';
import { Colors, FontFamily, Spacing, BorderRadius } from '../theme';

const SAMPLE_PLANETS = [
  { name: 'Kepler-452b', type: 'Super-Earth', distance: '1,400 ly', temp: '265 K', star: 'G-type' },
  { name: 'Proxima Centauri b', type: 'Terrestrial', distance: '4.2 ly', temp: '234 K', star: 'M-dwarf' },
  { name: 'TRAPPIST-1e', type: 'Earth-sized', distance: '39 ly', temp: '251 K', star: 'Ultra-cool' },
  { name: 'TOI-700d', type: 'Rocky', distance: '101 ly', temp: '269 K', star: 'M-dwarf' },
  { name: 'K2-18b', type: 'Mini-Neptune', distance: '124 ly', temp: '265 K', star: 'K-type' },
  { name: 'LHS 1140b', type: 'Super-Earth', distance: '41 ly', temp: '230 K', star: 'M-dwarf' },
];

const FILTERS = ['All', 'Terrestrial', 'Super-Earth', 'Gas Giant', 'Mini-Neptune'];

export default function SearchScreen() {
  const [query, setQuery] = React.useState('');
  const [activeFilter, setActiveFilter] = React.useState('All');
  const inputAnim = React.useRef(new Animated.Value(0)).current;

  const filtered = SAMPLE_PLANETS.filter((p) => {
    const matchQuery = p.name.toLowerCase().includes(query.toLowerCase());
    const matchFilter = activeFilter === 'All' || p.type.includes(activeFilter);
    return matchQuery && matchFilter;
  });

  const handleFocus = () => {
    Animated.timing(inputAnim, { toValue: 1, duration: 250, useNativeDriver: false }).start();
  };
  const handleBlur = () => {
    Animated.timing(inputAnim, { toValue: 0, duration: 250, useNativeDriver: false }).start();
  };

  const borderColor = inputAnim.interpolate({
    inputRange: [0, 1],
    outputRange: [Colors.border, Colors.accent],
  });

  return (
    <View style={styles.container}>
      <Text style={styles.heading}>PLANET SEARCH</Text>
      <Text style={styles.subheading}>Browse the confirmed exoplanet catalogue</Text>

      {/* Search bar */}
      <Animated.View style={[styles.inputWrapper, { borderColor }]}>
        <Text style={styles.searchIcon}>🔍</Text>
        <TextInput
          style={styles.input}
          placeholder="Search exoplanets..."
          placeholderTextColor={Colors.textMuted}
          value={query}
          onChangeText={setQuery}
          onFocus={handleFocus}
          onBlur={handleBlur}
          selectionColor={Colors.accent}
        />
      </Animated.View>

      {/* Filter chips */}
      <ScrollView
        horizontal
        showsHorizontalScrollIndicator={false}
        style={styles.filtersScroll}
        contentContainerStyle={styles.filtersContent}
      >
        {FILTERS.map((f) => (
          <TouchableOpacity
            key={f}
            style={[styles.chip, activeFilter === f && styles.chipActive]}
            onPress={() => setActiveFilter(f)}
          >
            <Text style={[styles.chipText, activeFilter === f && styles.chipTextActive]}>
              {f}
            </Text>
          </TouchableOpacity>
        ))}
      </ScrollView>

      {/* Results */}
      <ScrollView style={styles.list} showsVerticalScrollIndicator={false}>
        {filtered.map((planet) => (
          <TouchableOpacity key={planet.name} activeOpacity={0.75} style={styles.card}>
            <View style={styles.cardLeft}>
              <View style={styles.planetDot} />
              <View>
                <Text style={styles.planetName}>{planet.name}</Text>
                <Text style={styles.planetType}>{planet.type}</Text>
              </View>
            </View>
            <View style={styles.cardRight}>
              <Text style={styles.cardMeta}>⭐ {planet.star}</Text>
              <Text style={styles.cardMeta}>📏 {planet.distance}</Text>
              <Text style={styles.cardMeta}>🌡 {planet.temp}</Text>
            </View>
          </TouchableOpacity>
        ))}
        {filtered.length === 0 && (
          <View style={styles.empty}>
            <Text style={styles.emptyIcon}>🌌</Text>
            <Text style={styles.emptyText}>No planets found</Text>
          </View>
        )}
        <View style={{ height: 80 }} />
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: Colors.background,
    paddingTop: 60,
    paddingHorizontal: Spacing.lg,
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
    marginBottom: Spacing.md,
  },
  inputWrapper: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: Colors.surface,
    borderWidth: 1,
    borderRadius: BorderRadius.md,
    paddingHorizontal: Spacing.md,
    marginBottom: Spacing.md,
  },
  searchIcon: { fontSize: 16, marginRight: Spacing.sm },
  input: {
    flex: 1,
    height: 48,
    fontFamily: FontFamily.regular,
    fontSize: 14,
    color: Colors.text,
  },
  filtersScroll: { maxHeight: 44, marginBottom: Spacing.md },
  filtersContent: { gap: Spacing.sm, paddingRight: Spacing.md },
  chip: {
    borderWidth: 1,
    borderColor: Colors.border,
    borderRadius: BorderRadius.full,
    paddingHorizontal: Spacing.md,
    paddingVertical: 6,
    backgroundColor: Colors.surface,
  },
  chipActive: {
    borderColor: Colors.accent,
    backgroundColor: Colors.accentDim,
  },
  chipText: {
    fontFamily: FontFamily.regular,
    fontSize: 11,
    color: Colors.textMuted,
  },
  chipTextActive: { color: Colors.accent },
  list: { flex: 1 },
  card: {
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
  cardLeft: { flexDirection: 'row', alignItems: 'center', gap: Spacing.sm, flex: 1 },
  planetDot: {
    width: 28,
    height: 28,
    borderRadius: 14,
    backgroundColor: Colors.accentDim,
    borderWidth: 1.5,
    borderColor: Colors.accent,
  },
  planetName: {
    fontFamily: FontFamily.bold,
    fontSize: 13,
    color: Colors.text,
  },
  planetType: {
    fontFamily: FontFamily.regular,
    fontSize: 10,
    color: Colors.textMuted,
    marginTop: 2,
  },
  cardRight: { alignItems: 'flex-end', gap: 2 },
  cardMeta: {
    fontFamily: FontFamily.regular,
    fontSize: 10,
    color: Colors.textMuted,
  },
  empty: { alignItems: 'center', marginTop: 60 },
  emptyIcon: { fontSize: 48, marginBottom: Spacing.md },
  emptyText: { fontFamily: FontFamily.regular, color: Colors.textMuted, fontSize: 14 },
});
