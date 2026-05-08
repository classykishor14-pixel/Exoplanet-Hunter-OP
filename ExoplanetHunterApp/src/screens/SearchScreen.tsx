import React, { useEffect, useState, useRef } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TextInput,
  ScrollView,
  TouchableOpacity,
  Animated,
  ActivityIndicator,
  Platform,
  FlatList,
  Keyboard,
  Image,
} from 'react-native';
import { useNavigation } from '@react-navigation/native';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { Colors, FontFamily, Spacing, BorderRadius } from '../theme';
import { getPlanetDetails, autocompletePlanets, getRecommendations } from '../services/api';
import { Recommendation } from '../services/api';

// NOTE: If testing on a physical device, change this to your computer's local IP address (e.g., 'http://192.168.1.100:8000')
const API_BASE_URL = Platform.OS === 'android' ? 'http://10.0.2.2:8000' : 'http://127.0.0.1:8000';

const POPULAR_PLANETS = [
  'Kepler-442 b',
  'TRAPPIST-1 e',
  'Proxima Centauri b',
  'Kepler-186 f',
  'K2-18 b',
  'LHS 1140 b',
];

const DISCOVER_CATEGORIES = ['All', 'Habitable', 'Gas Giants', 'Rocky', 'Lava Worlds'];

const SEARCH_HISTORY_KEY = '@search_history';

interface DiscoverCardProps {
  recommendation: Recommendation;
  onPress: () => void;
}

const DiscoverCard: React.FC<DiscoverCardProps> = ({ recommendation, onPress }) => {
  const scaleAnim = useRef(new Animated.Value(1)).current;

  const handlePressIn = () => {
    Animated.spring(scaleAnim, {
      toValue: 0.95,
      useNativeDriver: true,
    }).start();
  };

  const handlePressOut = () => {
    Animated.spring(scaleAnim, {
      toValue: 1,
      useNativeDriver: true,
    }).start();
  };

  return (
    <Animated.View style={{ transform: [{ scale: scaleAnim }] }}>
      <TouchableOpacity
        style={styles.discoverCard}
        onPress={onPress}
        onPressIn={handlePressIn}
        onPressOut={handlePressOut}
        activeOpacity={0.9}
      >
        <Image
          source={{ uri: recommendation.thumbnail_url }}
          style={styles.cardThumbnail}
          resizeMode="cover"
        />
        <View style={styles.cardOverlay}>
          <View style={styles.cardHeader}>
            <Text style={styles.cardEmoji}>{recommendation.type_emoji}</Text>
            <Text style={styles.cardName}>{recommendation.name}</Text>
          </View>
          <Text style={styles.cardHost}>Host: {recommendation.host_star}</Text>
          <Text style={styles.cardDescription} numberOfLines={2}>
            {recommendation.description}
          </Text>
        </View>
      </TouchableOpacity>
    </Animated.View>
  );
};

export default function SearchScreen() {
  const navigation = useNavigation();
  const [query, setQuery] = useState('');
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const [searchHistory, setSearchHistory] = useState<string[]>([]);
  const [searchResults, setSearchResults] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [showDropdown, setShowDropdown] = useState(false);
  const [recommendations, setRecommendations] = useState<Recommendation[]>([]);
  const [activeCategory, setActiveCategory] = useState('All');

  const inputAnim = useRef(new Animated.Value(0)).current;
  const dropdownAnim = useRef(new Animated.Value(0)).current;

  // Load search history and recommendations on mount
  useEffect(() => {
    loadSearchHistory();
    loadRecommendations();
  }, []);

  const loadRecommendations = async () => {
    try {
      const recs = await getRecommendations();
      setRecommendations(recs);
    } catch (error) {
      console.error('Error loading recommendations:', error);
    }
  };

  // Autocomplete as user types
  useEffect(() => {
    if (query.length >= 2) {
      fetchSuggestions(query);
    } else {
      setSuggestions([]);
      setShowDropdown(false);
    }
  }, [query]);

  const loadSearchHistory = async () => {
    try {
      const history = await AsyncStorage.getItem(SEARCH_HISTORY_KEY);
      if (history) {
        setSearchHistory(JSON.parse(history));
      }
    } catch (error) {
      console.error('Error loading search history:', error);
    }
  };

  const saveSearchHistory = async (newHistory: string[]) => {
    try {
      await AsyncStorage.setItem(SEARCH_HISTORY_KEY, JSON.stringify(newHistory));
      setSearchHistory(newHistory);
    } catch (error) {
      console.error('Error saving search history:', error);
    }
  };

  const addToHistory = (searchTerm: string) => {
    const updatedHistory = [searchTerm, ...searchHistory.filter(item => item !== searchTerm)].slice(0, 10);
    saveSearchHistory(updatedHistory);
  };

  const fetchSuggestions = async (searchQuery: string) => {
    try {
      const result = await autocompletePlanets(searchQuery);
      setSuggestions(result);
      setShowDropdown(result.length > 0);
      Animated.timing(dropdownAnim, { toValue: result.length > 0 ? 1 : 0, duration: 200, useNativeDriver: false }).start();
    } catch (error) {
      console.error('Error fetching suggestions:', error);
      setSuggestions([]);
      setShowDropdown(false);
    }
  };

  const handleSearch = async (searchTerm?: string) => {
    const term = searchTerm || query;
    if (!term.trim()) return;

    setLoading(true);
    setError('');
    setShowDropdown(false);
    Keyboard.dismiss();

    try {
      const data = await getPlanetDetails(term);
      setSearchResults([data]);
      addToHistory(term);
      setQuery(term);
      // Navigate to details screen
      navigation.navigate('PlanetDetails', { result: data });
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Planet not found');
      setSearchResults([]);
    } finally {
      setLoading(false);
    }
  };

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

  const shadowOpacity = inputAnim.interpolate({
    inputRange: [0, 1],
    outputRange: [0, 0.8],
  });

  const filteredRecommendations = recommendations.filter((rec) => {
    if (activeCategory === 'All') return true;
    return rec.category === activeCategory;
  });

  return (
    <View style={styles.container}>
      <Text style={styles.heading}>PLANET SEARCH</Text>
      <Text style={styles.subheading}>Discover exoplanets from NASA archives</Text>

      {/* Search bar */}
      <View style={styles.inputContainer}>
        <Animated.View style={[
          styles.inputWrapper,
          {
            borderColor,
            shadowOpacity,
            shadowColor: Colors.accent,
            shadowRadius: 10,
            elevation: 10,
          }
        ]}>
          <Text style={styles.searchIcon}>🔍</Text>
          <TextInput
            style={styles.input}
            placeholder="Search exoplanets..."
            placeholderTextColor={Colors.textMuted}
            value={query}
            onChangeText={(text) => {
              setQuery(text);
              if (text === '') {
                setSearchResults([]);
                setError('');
              }
            }}
            onSubmitEditing={() => handleSearch()}
            onFocus={handleFocus}
            onBlur={handleBlur}
            selectionColor={Colors.accent}
            returnKeyType="search"
            autoCapitalize="none"
            autoCorrect={false}
          />
        </Animated.View>

        {/* Autocomplete Dropdown */}
        {showDropdown && (
          <Animated.View style={[styles.dropdown, { height: dropdownHeight }]}>
            <FlatList
              data={suggestions}
              keyExtractor={(item) => item}
              renderItem={({ item }) => (
                <TouchableOpacity
                  style={styles.dropdownItem}
                  onPress={() => {
                    setQuery(item);
                    handleSearch(item);
                  }}
                >
                  <Text style={styles.dropdownText}>{item}</Text>
                </TouchableOpacity>
              )}
              showsVerticalScrollIndicator={false}
            />
          </Animated.View>
        )}
      </View>

      {/* Discover Panel */}
      <View style={styles.section}>
        <View style={styles.discoverHeader}>
          <Text style={styles.sectionTitle}>DISCOVER</Text>
          <ScrollView
            horizontal
            showsHorizontalScrollIndicator={false}
            style={styles.categoryScroll}
            contentContainerStyle={styles.categoryContent}
          >
            {DISCOVER_CATEGORIES.map((category) => (
              <TouchableOpacity
                key={category}
                style={[
                  styles.categoryChip,
                  activeCategory === category && styles.categoryChipActive
                ]}
                onPress={() => setActiveCategory(category)}
              >
                <Text style={[
                  styles.categoryChipText,
                  activeCategory === category && styles.categoryChipTextActive
                ]}>
                  {category}
                </Text>
              </TouchableOpacity>
            ))}
          </ScrollView>
        </View>
        
        <FlatList
          horizontal
          showsHorizontalScrollIndicator={false}
          data={filteredRecommendations}
          keyExtractor={(item) => item.name}
          renderItem={({ item }) => (
            <DiscoverCard
              recommendation={item}
              onPress={() => handleSearch(item.name)}
            />
          )}
          contentContainerStyle={styles.carouselContent}
          ItemSeparatorComponent={() => <View style={{ width: Spacing.sm }} />}
        />
      </View>

      {/* Popular Searches */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>POPULAR SEARCHES</Text>
        <View style={styles.chipsContainer}>
          {POPULAR_PLANETS.map((planet) => (
            <TouchableOpacity
              key={planet}
              style={styles.chip}
              onPress={() => handleSearch(planet)}
            >
              <Text style={styles.chipText}>{planet}</Text>
            </TouchableOpacity>
          ))}
        </View>
      </View>

      {/* Search History */}
      {searchHistory.length > 0 && (
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>RECENT SEARCHES</Text>
          <View style={styles.chipsContainer}>
            {searchHistory.slice(0, 6).map((item) => (
              <TouchableOpacity
                key={item}
                style={[styles.chip, styles.historyChip]}
                onPress={() => handleSearch(item)}
              >
                <Text style={styles.chipText}>{item}</Text>
              </TouchableOpacity>
            ))}
          </View>
        </View>
      )}

      {/* Results */}
      <ScrollView style={styles.list} showsVerticalScrollIndicator={false}>
        {loading && (
          <ActivityIndicator size="large" color={Colors.accent} style={{ marginTop: 40 }} />
        )}

        {error !== '' && !loading && (
          <View style={styles.empty}>
            <Text style={styles.emptyIcon}>⚠️</Text>
            <Text style={styles.emptyText}>{error}</Text>
          </View>
        )}

        {/* Search Results */}
        {!loading && searchResults.length > 0 && (
          <View>
            <Text style={styles.sectionTitle}>SEARCH RESULT</Text>
            {searchResults.map((data, index) => (
              <TouchableOpacity
                key={index}
                style={styles.card}
                onPress={() => navigation.navigate('PlanetDetails', { result: data })}
                activeOpacity={0.7}
              >
                <View style={styles.cardLeft}>
                  <View style={[
                    styles.planetDot,
                    data.analysis.is_in_habitable_zone ? { borderColor: Colors.success } : {}
                  ]} />
                  <View style={{ flex: 1, paddingRight: 10 }}>
                    <Text style={styles.planetName}>{data.planet}</Text>
                    <Text style={styles.planetType}>
                      {data.analysis.planet_type}
                    </Text>
                    <Text style={styles.planetDesc} numberOfLines={2}>
                      {data.analysis.composition_estimate}
                    </Text>
                  </View>
                </View>
                <View style={styles.cardRight}>
                  {data.analysis.is_in_habitable_zone && (
                    <Text style={[styles.cardMeta, { color: Colors.success }]}>🌿 Habitable Zone</Text>
                  )}
                  {data.analysis.density_g_cm3 && (
                    <Text style={styles.cardMeta}>⚖️ {data.analysis.density_g_cm3} g/cm³</Text>
                  )}
                </View>
              </TouchableOpacity>
            ))}
          </View>
        )}

        <View style={{ height: 80 }} />
      </ScrollView>
    </View>
  );
}
    setError('');
    try {
      const res = await fetch(`${API_BASE_URL}/api/planet/${encodeURIComponent(query)}/details`);
      const data = await res.json();
      
      if (res.ok && data.planet) {
        setSearchResults([data]);
      } else {
        setSearchResults([]);
        setError(data.detail || 'Planet not found in the archive.');
      }
    } catch (err) {
      setError('Network error. Is the backend server running?');
      setSearchResults([]);
    } finally {
      setLoading(false);
    }
  };

  const filteredRecs = recommendations.filter((p) => {
    if (activeFilter === 'All') return true;
    return p.description.toLowerCase().includes(activeFilter.toLowerCase());
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
      <Text style={styles.subheading}>Browse the confirmed exoplanet catalogue via NASA TAP</Text>

      {/* Search bar */}
      <Animated.View style={[styles.inputWrapper, { borderColor }]}>
        <Text style={styles.searchIcon}>🔍</Text>
        <TextInput
          style={styles.input}
          placeholder="Search e.g. 'Kepler-442 b'..."
          placeholderTextColor={Colors.textMuted}
          value={query}
          onChangeText={(text) => {
            setQuery(text);
            if (text === '') setSearchResults([]); // clear results if empty
          }}
          onSubmitEditing={handleSearch}
          onFocus={handleFocus}
          onBlur={handleBlur}
          selectionColor={Colors.accent}
          returnKeyType="search"
        />
      </Animated.View>

      {/* Show filters only if we are browsing recommendations */}
      {query === '' && (
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
      )}

      {/* Results */}
      <ScrollView style={styles.list} showsVerticalScrollIndicator={false}>
        
        {loading && (
          <ActivityIndicator size="large" color={Colors.accent} style={{ marginTop: 40 }} />
        )}

        {error !== '' && !loading && (
          <View style={styles.empty}>
            <Text style={styles.emptyIcon}>⚠️</Text>
            <Text style={styles.emptyText}>{error}</Text>
          </View>
        )}

        {/* Live Search Results */}
        {!loading && searchResults.length > 0 && query !== '' && (
          <View>
            <Text style={styles.sectionTitle}>LIVE ANALYSIS</Text>
            {searchResults.map((data, index) => (
              <View key={index} style={styles.card}>
                <View style={styles.cardLeft}>
                  <View style={[
                    styles.planetDot, 
                    data.analysis.is_in_habitable_zone ? { borderColor: Colors.success } : {}
                  ]} />
                  <View style={{ flex: 1, paddingRight: 10 }}>
                    <Text style={styles.planetName}>{data.planet}</Text>
                    <Text style={styles.planetType}>
                      {data.analysis.planet_type}
                    </Text>
                    <Text style={styles.planetDesc} numberOfLines={2}>
                      {data.analysis.composition_estimate}
                    </Text>
                  </View>
                </View>
                <View style={styles.cardRight}>
                  {data.analysis.is_in_habitable_zone && (
                    <Text style={[styles.cardMeta, { color: Colors.success }]}>🌿 Habitable Zone</Text>
                  )}
                  {data.analysis.density_g_cm3 && (
                    <Text style={styles.cardMeta}>⚖️ {data.analysis.density_g_cm3} g/cm³</Text>
                  )}
                </View>
              </View>
            ))}
          </View>
        )}

        {/* Famous Recommendations */}
        {!loading && searchResults.length === 0 && query === '' && filteredRecs.length > 0 && (
          <View>
            <Text style={styles.sectionTitle}>FAMOUS DISCOVERIES</Text>
            {filteredRecs.map((planet) => (
              <TouchableOpacity key={planet.name} activeOpacity={0.75} style={styles.card}>
                <View style={styles.cardLeft}>
                  <View style={styles.planetDot} />
                  <View style={{ flex: 1, paddingRight: 10 }}>
                    <Text style={styles.planetName}>{planet.name}</Text>
                    <Text style={styles.planetDesc} numberOfLines={3}>{planet.description}</Text>
                  </View>
                </View>
              </TouchableOpacity>
            ))}
          </View>
        )}

        {!loading && searchResults.length === 0 && query === '' && filteredRecs.length === 0 && (
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
  inputContainer: {
    marginBottom: Spacing.md,
    zIndex: 10,
  },
  inputWrapper: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(13, 21, 48, 0.8)',
    borderWidth: 1,
    borderRadius: BorderRadius.md,
    paddingHorizontal: Spacing.md,
    backdropFilter: 'blur(10px)',
  },
  searchIcon: { fontSize: 16, marginRight: Spacing.sm },
  input: {
    flex: 1,
    height: 48,
    fontFamily: FontFamily.regular,
    fontSize: 14,
    color: Colors.text,
  },
  dropdown: {
    position: 'absolute',
    top: '100%',
    left: 0,
    right: 0,
    backgroundColor: 'rgba(8, 13, 26, 0.95)',
    borderWidth: 1,
    borderColor: Colors.border,
    borderTopWidth: 0,
    borderRadius: BorderRadius.md,
    marginTop: -BorderRadius.md,
    overflow: 'hidden',
    backdropFilter: 'blur(10px)',
  },
  dropdownItem: {
    paddingHorizontal: Spacing.md,
    paddingVertical: Spacing.sm,
    borderBottomWidth: 1,
    borderBottomColor: Colors.border,
  },
  dropdownText: {
    fontFamily: FontFamily.regular,
    fontSize: 14,
    color: Colors.text,
  },
  section: {
    marginBottom: Spacing.lg,
  },
  discoverHeader: {
    marginBottom: Spacing.md,
  },
  sectionTitle: {
    fontFamily: FontFamily.bold,
    fontSize: 12,
    color: Colors.accentDim,
    letterSpacing: 2,
    marginBottom: Spacing.sm,
  },
  categoryScroll: {
    maxHeight: 36,
    marginTop: Spacing.xs,
  },
  categoryContent: {
    gap: Spacing.sm,
    paddingRight: Spacing.md,
  },
  categoryChip: {
    borderWidth: 1,
    borderColor: Colors.border,
    borderRadius: BorderRadius.full,
    paddingHorizontal: Spacing.md,
    paddingVertical: 6,
    backgroundColor: 'rgba(13, 21, 48, 0.6)',
    justifyContent: 'center',
  },
  categoryChipActive: {
    borderColor: Colors.accent,
    backgroundColor: 'rgba(0, 255, 255, 0.1)',
  },
  categoryChipText: {
    fontFamily: FontFamily.regular,
    fontSize: 11,
    color: Colors.textMuted,
  },
  categoryChipTextActive: {
    color: Colors.accent,
  },
  carouselContent: {
    paddingRight: Spacing.md,
  },
  discoverCard: {
    width: 280,
    height: 200,
    borderRadius: BorderRadius.md,
    backgroundColor: 'rgba(5, 12, 32, 0.88)',
    borderWidth: 1,
    borderColor: Colors.accent,
    overflow: 'hidden',
    shadowColor: Colors.accent,
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
    elevation: 8,
  },
  cardThumbnail: {
    width: '100%',
    height: 120,
  },
  cardOverlay: {
    flex: 1,
    padding: Spacing.sm,
    justifyContent: 'space-between',
  },
  cardHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: Spacing.xs,
  },
  cardEmoji: {
    fontSize: 18,
  },
  cardName: {
    fontFamily: FontFamily.bold,
    fontSize: 14,
    color: Colors.text,
    flex: 1,
  },
  cardHost: {
    fontFamily: FontFamily.regular,
    fontSize: 10,
    color: Colors.accent,
    marginBottom: 2,
  },
  cardDescription: {
    fontFamily: FontFamily.regular,
    fontSize: 10,
    color: Colors.textMuted,
    lineHeight: 14,
  },
  chipsContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: Spacing.sm,
  },
  chip: {
    backgroundColor: 'rgba(0, 255, 255, 0.1)',
    borderWidth: 1,
    borderColor: Colors.accent,
    borderRadius: BorderRadius.full,
    paddingHorizontal: Spacing.md,
    paddingVertical: 8,
    shadowColor: Colors.accent,
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.3,
    shadowRadius: 5,
    elevation: 5,
  },
  historyChip: {
    backgroundColor: 'rgba(26, 40, 64, 0.8)',
    borderColor: Colors.border,
  },
  chipText: {
    fontFamily: FontFamily.regular,
    fontSize: 11,
    color: Colors.accent,
    letterSpacing: 0.5,
  },
  list: { flex: 1 },
  card: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    backgroundColor: 'rgba(13, 21, 48, 0.8)',
    borderWidth: 1,
    borderColor: Colors.border,
    borderRadius: BorderRadius.md,
    padding: Spacing.md,
    marginBottom: Spacing.sm,
    backdropFilter: 'blur(10px)',
    shadowColor: Colors.accent,
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.2,
    shadowRadius: 8,
    elevation: 8,
  },
  cardLeft: { flexDirection: 'row', alignItems: 'flex-start', gap: Spacing.sm, flex: 1 },
  planetDot: {
    width: 28,
    height: 28,
    borderRadius: 14,
    backgroundColor: Colors.accentDim,
    borderWidth: 1.5,
    borderColor: Colors.accent,
    marginTop: 2,
  },
  planetName: {
    fontFamily: FontFamily.bold,
    fontSize: 14,
    color: Colors.text,
    marginBottom: 2,
  },
  planetType: {
    fontFamily: FontFamily.bold,
    fontSize: 10,
    color: Colors.accent,
    marginBottom: 4,
  },
  planetDesc: {
    fontFamily: FontFamily.regular,
    fontSize: 10,
    color: Colors.textMuted,
    lineHeight: 14,
  },
  cardRight: { alignItems: 'flex-end', gap: 4, justifyContent: 'center' },
  cardMeta: {
    fontFamily: FontFamily.regular,
    fontSize: 9,
    color: Colors.textMuted,
  },
  empty: { alignItems: 'center', marginTop: 60 },
  emptyIcon: { fontSize: 48, marginBottom: Spacing.md },
  emptyText: { fontFamily: FontFamily.regular, color: Colors.textMuted, fontSize: 14 },
});
