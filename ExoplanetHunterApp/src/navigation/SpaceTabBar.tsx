import React from 'react';
import { View, Text, TouchableOpacity, StyleSheet, Animated } from 'react-native';
import { BottomTabBarProps } from '@react-navigation/bottom-tabs';
import { Colors, FontFamily } from '../theme';

const TABS = [
  { key: 'Home', label: 'HOME', icon: '🏠' },
  { key: 'Search', label: 'SEARCH', icon: '🔍' },
  { key: 'ARCamera', label: 'AR CAM', icon: '📷' },
  { key: 'Profile', label: 'PROFILE', icon: '👤' },
] as const;

export default function SpaceTabBar({ state, navigation }: BottomTabBarProps) {
  const anims = React.useRef(TABS.map(() => new Animated.Value(0))).current;

  const handlePress = (index: number, routeName: string, isFocused: boolean) => {
    Animated.spring(anims[index], {
      toValue: 1,
      friction: 4,
      tension: 180,
      useNativeDriver: true,
    }).start(() => {
      Animated.spring(anims[index], {
        toValue: 0,
        friction: 6,
        tension: 200,
        useNativeDriver: true,
      }).start();
    });

    if (!isFocused) {
      navigation.navigate(routeName);
    }
  };

  return (
    <View style={styles.container}>
      {/* Glow border on top */}
      <View style={styles.topGlow} />

      {TABS.map((tab, i) => {
        const isFocused = state.index === i;
        const scale = anims[i].interpolate({
          inputRange: [0, 1],
          outputRange: [1, 1.25],
        });

        return (
          <TouchableOpacity
            key={tab.key}
            style={styles.tab}
            onPress={() => handlePress(i, tab.key, isFocused)}
            activeOpacity={0.7}
            accessibilityRole="button"
            accessibilityLabel={tab.label}
          >
            {isFocused && <View style={styles.activeIndicator} />}
            <Animated.Text
              style={[styles.icon, { transform: [{ scale }] }]}
            >
              {tab.icon}
            </Animated.Text>
            <Text
              style={[
                styles.label,
                isFocused ? styles.labelActive : styles.labelInactive,
              ]}
            >
              {tab.label}
            </Text>
          </TouchableOpacity>
        );
      })}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    backgroundColor: Colors.tabBar,
    height: 70,
    paddingBottom: 8,
    borderTopWidth: 1,
    borderTopColor: Colors.border,
    position: 'relative',
  },
  topGlow: {
    position: 'absolute',
    top: -1,
    left: 0,
    right: 0,
    height: 1,
    backgroundColor: Colors.accent,
    opacity: 0.4,
    shadowColor: Colors.accent,
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 1,
    shadowRadius: 8,
    elevation: 8,
  },
  tab: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'flex-end',
    paddingBottom: 4,
    position: 'relative',
  },
  activeIndicator: {
    position: 'absolute',
    top: 0,
    width: 32,
    height: 2,
    borderRadius: 1,
    backgroundColor: Colors.accent,
    shadowColor: Colors.accent,
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 1,
    shadowRadius: 6,
    elevation: 6,
  },
  icon: {
    fontSize: 22,
    marginBottom: 2,
  },
  label: {
    fontFamily: FontFamily.regular,
    fontSize: 8,
    letterSpacing: 1,
  },
  labelActive: {
    color: Colors.tabActive,
  },
  labelInactive: {
    color: Colors.tabInactive,
  },
});
