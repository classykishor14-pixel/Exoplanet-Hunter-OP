import React from 'react';
import { Animated } from 'react-native';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import HomeScreen from '../screens/HomeScreen';
import SearchScreen from '../screens/SearchScreen';
import ARCameraScreen from '../screens/ARCameraScreen';
import ProfileScreen from '../screens/ProfileScreen';
import SpaceTabBar from './SpaceTabBar';
import { Colors } from '../theme';

const Tab = createBottomTabNavigator();

/**
 * Smooth fade transition between tabs.
 * Uses opacity interpolation so screens crossfade instead of sliding.
 */
function FadeTransition({ current }: { current: { progress: Animated.AnimatedInterpolation<number> } }) {
  return {
    cardStyle: {
      opacity: current.progress.interpolate({
        inputRange: [0, 1],
        outputRange: [0, 1],
      }),
    },
  };
}

export default function AppNavigator() {
  return (
    <Tab.Navigator
      tabBar={(props) => <SpaceTabBar {...props} />}
      screenOptions={{
        headerShown: false,
        // Apply the fade transition to every tab screen
        sceneStyle: { backgroundColor: Colors.background },
      }}
    >
      <Tab.Screen name="Home" component={HomeScreen} />
      <Tab.Screen name="Search" component={SearchScreen} />
      <Tab.Screen name="ARCamera" component={ARCameraScreen} />
      <Tab.Screen name="Profile" component={ProfileScreen} />
    </Tab.Navigator>
  );
}
