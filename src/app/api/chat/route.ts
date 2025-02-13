import { openai } from "@ai-sdk/openai";
import { embed, generateObject, streamText, tool, type CoreMessage } from "ai";
import { z } from "zod";
import { supabase } from "../../../lib/supabase";

export interface Route {
  id: string;
  name: string;
  quality?: string;
  type?: string;
  grade?: string;
  area_name?: string;
  description?: string;
  length?: number;
  bolts?: number;
  createdAt?: Date;
  updatedAt?: Date;
  areaId: string;
}

interface ChatRequest {
  messages: CoreMessage[];
}

// Add these interfaces for weather data typing
interface WeatherReading {
  dt: number;
  dt_txt: string;
  main: {
    temp: number;
    feels_like: number;
    humidity: number;
  };
  weather: Array<{
    description: string;
  }>;
  wind: {
    speed: number;
  };
}

interface WeatherResponse {
  list: WeatherReading[];
}

export async function POST(req: Request) {
  const { messages }: ChatRequest = await req.json();

  const result = streamText({
    model: openai("gpt-4o-mini"),
    system: `You are "cactux", an expert climbing guide for Guadalcazar (small climbing town in San Luis Potosi, Mexico) with a notoriously sarcastic attitude. You have access to a comprehensive database of local climbing information.

    === Climbing Guide Rules ===
    - Knowledge Base includes:
    • Climbing routes (names, descriptions, grades, types, lengths, bolts, quality)
    • Local amenities (restaurants, accommodations)
    • Climbing zones information
    - Core Behaviors:
    1. Always use RAG (Retrieval-Augmented Generation) for initial data lookup.
    2. Act immediately without asking for permission.
    3. Provide direct and accurate responses based solely on available data.
    4. Clearly indicate when information is not found in the database.
    5. Always process and incorporate the data from the retrieveRelevantClimbingData tool.
    6. Format your responses with the appropriate climbing data.
    7. If the user's query is in Spanish, reply in Spanish; otherwise, use English.
    8. When discussing routes, consider their quality ratings:
        - 80-100: Outstanding/classic routes.
        - 60-79: Very good routes.
        - 40-59: Good routes.
        - Below 40: Less recommended routes.
    - Strictly provide information on Guadalcazar climbing and related local services. For non-climbing topics, clarify your focus on climbing.

    === Personality Rules for Cactux ===
    Core Identity: You're "Cactux," a jaded climbing guide who's seen every climbing ego trip possible and isn't afraid to call them out.

    Tone & Attitude:

    Use desert-dry sarcasm and brutal honesty
    Never apologize - if someone's feelings get hurt, that's their beta to figure out
    Respond to basic questions with mock exasperation but still provide accurate info
    Include climbing-specific burns like "Nice project, bro. Did you flash it? Oh wait, you're still working on V2."

    **Cultural Commentary:**

    -Mercilessly mock common climber stereotypes:

    ---The grade-obsessed climber who can't enjoy anything below their project level
    ---The "actually I downgraded that route" person
    ---Instagram climbers who spend more time on filters than sending
    ---People who claim they "almost had it" after falling on the first move
    ---Those who blame conditions/temperature/humidity for every failed attempt

    -Knowledge Base Focus:

    ---Only provide beta about Guadalcazar climbing areas
    ---If asked about other topics, respond with "Look, unless this somehow relates to sending in Guadalcazar, I really don't care. Have you tried our 13c routes?"

    -Response Style:

    ---Start responses with sarcastic observations about climbing culture
    ---Use climbing slang ironically
    ---Mock the obsession with gear while still giving correct gear advice
    ---Call out spray-lords and beta-sprayers while being one yourself

    -Running Jokes:

    ---Constantly reference how "back in my day" climbs were harder
    ---Mock the evolution of crash pad thickness
    ---Make fun of how every climber thinks their project is "definitely soft for the grade"
    ---Sarcastically praise ridiculous warmup routines

    -Teaching Style:

    ---Give accurate beta wrapped in sarcasm
    ---Mock common beginner mistakes while explaining how to avoid them
    ---Roast poor technique while demonstrating correct form
    ---Always emphasize safety points (the one thing Cactux is serious about)

    -Sample Response Example:
        "Oh great, another 'crusher' asking about our 6a routes. slow clap Let me guess, you'll probably tell me it feels more like 5+ because you 'climb 8a back home.' Look, if you must know, start with the crimp on the left, which - shocking revelation - requires actual technique, not just your campus board gains. And please, for the love of chalk, clip all the quickdraws. I know it's not as cool for your Instagram story, but I'd prefer not to do paperwork today."
    `,
    messages,
    toolChoice: "auto",
    tools: {
      identifyZone: tool({
        description:
          "Identify which climbing zone in Guadalcazar the user is asking about",
        parameters: z.object({
          userQuery: z
            .string()
            .describe(
              "The user's query about Guadalcazar climbing, returns the complete user query",
            ),
        }),
        execute: async ({ userQuery }) => {
          const result = await generateObject({
            model: openai("gpt-4o-mini"),
            system: `You are a climbing zone identifier for Guadalcazar. Your task is to identify which zone the user is asking about.

              Available zones and their alternative names:
              - Gruta de las Candelas (also known as: Las Candelas, Candelas)
              - Joya del Salitre (also known as: Salitre, El Salitre)
              - Panales
              - San Cayetano (also known as: San caye, Cayetano)
              - Zelda (also known as: Cuevas cuatas)

              IMPORTANT: Respond ONLY with one of these exact values:
              - "candelas"
              - "salitre"
              - "panales"
              - "cayetano"
              - "zelda"
              - "guadalcazar" (if query is about the general area)
              - null (if no zone can be confidently identified)

              When identifying San Cayetano or any of its variations, always return "cayetano".
              Handle typos and variations intelligently.
              `,
            prompt: `Identify the climbing zone from this query: ${userQuery}`,
            schema: z.object({
              zone: z.string(),
            }),
          });

          return result.object.zone;
        },
      }),
      weather: tool({
        description: "Get current and forecast weather for a climbing location",
        parameters: z.object({
          location: z.string().describe("The climbing zone to get weather for"),
        }),
        execute: async ({ location }) => {
          const normalizeLocation = (location: string): string => {
            if (!location) return "guadalcazar";

            const zoneMap: Record<string, string> = {
              "gruta de las candelas": "Gruta de las Candelas",
              "las candelas": "Gruta de las Candelas",
              candelas: "Gruta de las Candelas",
              "joya del salitre": "Joya del Salitre",
              "el salitre": "Joya del Salitre",
              salitre: "Joya del Salitre",
              panales: "Panales",
              "san cayetano": "San Cayetano",
              "san caye": "San Cayetano",
              cayetano: "San Cayetano",
              zelda: "Zelda",
              "cuevas cuatas": "Zelda",
              guadalcazar: "Guadalcazar",
            };

            const normalizedInput = location.toLowerCase().trim();
            return zoneMap[normalizedInput] || "Guadalcazar";
          };
          const normalizedLocation = normalizeLocation(location);
          console.log("normalizedLocation:", normalizedLocation);
          const tableName =
            normalizedLocation === "Guadalcazar" ? "place" : "area";
          try {
            // First get coordinates from database for the location
            const { data: locationData, error: locationError } = await supabase
              .from(tableName)
              .select("latitude, longitude")
              .ilike("name", normalizedLocation)
              .single();

            if (locationError || !locationData) {
              return {
                error: "Location coordinates not found in database",
              };
            }

            const { latitude, longitude } = locationData;

            // Call OpenWeatherMap API for 5 day forecast
            const OPENWEATHER_API_KEY = process.env.OPENWEATHER_API_KEY;
            const url = `https://api.openweathermap.org/data/2.5/forecast?lat=${latitude}&lon=${longitude}&appid=${OPENWEATHER_API_KEY}&units=metric`;
            const response = await fetch(url);
            if (!response.ok) {
              throw new Error("Weather API request failed");
            }

            const weatherData: WeatherResponse = await response.json();

            // Process and format the weather data with proper typing
            const forecast = weatherData.list
              .filter((reading: WeatherReading) =>
                reading.dt_txt.includes("12:00:00"),
              )
              .map((reading: WeatherReading) => ({
                date: new Date(reading.dt * 1000).toLocaleDateString(),
                temp: Math.round(reading.main.temp),
                feels_like: Math.round(reading.main.feels_like),
                conditions: reading.weather[0]?.description || "Desconocido",
                wind_speed: reading.wind.speed,
                humidity: reading.main.humidity,
              }));

            return {
              location,
              current: {
                temp: Math.round(weatherData.list[0]?.main.temp || 0),
                feels_like: Math.round(
                  weatherData.list[0]?.main.feels_like || 0,
                ),
                conditions:
                  weatherData.list[0]?.weather[0]?.description || "Desconocido",
                wind_speed: weatherData.list[0]?.wind.speed || 0,
                humidity: weatherData.list[0]?.main.humidity || 0,
              },
              forecast,
            };
          } catch (error) {
            console.error("Weather tool error:", error);
            return {
              error: "Failed to fetch weather data",
            };
          }
        },
      }),
      retrieveRelevantClimbingData: tool({
        description:
          "Search and retrieve relevant climbing information from the Guadalcazar database, including routes, grades, locations, and local amenities based on the user's query",
        parameters: z.object({
          userQuery: z
            .string()
            .describe(
              "The user's query about Guadalcazar climbing areas, routes, or local information",
            ),
          zone: z.string().describe("The zone to retrieve climbing data for"),
        }),
        execute: async ({ userQuery, zone }) => {
          const normalizeZone = (zone: string): string => {
            if (!zone) return "guadalcazar";

            const zoneMap: Record<string, string> = {
              "gruta de las candelas": "candelas",
              "las candelas": "candelas",
              candelas: "candelas",
              "joya del salitre": "salitre",
              "el salitre": "salitre",
              salitre: "salitre",
              panales: "panales",
              "san cayetano": "cayetano",
              "san caye": "cayetano",
              cayetano: "cayetano",
              zelda: "zelda",
              "cuevas cuatas": "zelda",
              guadalcazar: "guadalcazar",
            };

            const normalizedInput = zone.toLowerCase().trim();
            return zoneMap[normalizedInput] || "guadalcazar";
          };
          console.log("userQuery:", userQuery);
          console.log("zone:", zone);
          const normalizedZone = normalizeZone(zone);
          console.log("normalizedZone:", normalizedZone);

          const { embedding } = await embed({
            model: openai.embedding("text-embedding-3-small"),
            value: userQuery,
          });

          try {
            const { data, error } = await supabase.rpc("match_data", {
              query_embedding: embedding,
              match_count: 20,
              filter: normalizedZone,
            });

            if (error) throw error;

            // For the retrieveRelevantClimbingData tool, add interface for the matched data
            interface MatchedData {
              title: string;
              content: string;
            }

            // Update the data mapping
            const formatted_chunks = data.map(
              (doc: MatchedData) => `# ${doc.title}\n\n${doc.content}`,
            );

            // Join all chunks with a separator
            return formatted_chunks.join("\n\n---\n\n");
          } catch (error) {
            console.error("Error while retrieveRelevantClimbingData", error);
          }
        },
      }),
    },
  });

  return result.toDataStreamResponse();
}
