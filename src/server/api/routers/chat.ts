import { openai } from "@ai-sdk/openai";
import { generateObject } from "ai";
import { z } from "zod";
import { createTRPCRouter, publicProcedure } from "~/server/api/trpc";

export const chatRouter = createTRPCRouter({
  generateQuery: publicProcedure
    .input(z.object({ prompt: z.string() }))
    .mutation(async ({ input }) => {
      try {
        const result = await generateObject({
          model: openai("gpt-4o-mini"),
          system: `You are a SQL (postgres) expert specializing in climbing routes data from Guadalcazar, San Luis Potosi, Mexico. Your job is to help users write SQL queries to retrieve routes information. The table schema is as follows:

              "Route" (
                id          String    @id @default(uuid())
                name        String
                description String?
                grade       String?
                createdAt   DateTime?
                updatedAt   DateTime?
                areaId      String
                bolts       String?
                createdBy   String?
                length      String?
                quality     String?
                type        String?
              )

              "Area" (
                id          String   @id
                name        String   @unique
                description String?
                approach    String?
                ethic       String?
                createdAt   DateTime
                updatedAt   DateTime
              )

            Only retrieval queries are allowed.

            IMPORTANT: Table names must be quoted in double quotes in all queries (use "Route" and "Area", not route or area).
            Example: SELECT * FROM "Route" r JOIN "Area" a ON r."areaId" = a.id

            For string fields like name and type, use the ILIKE operator and convert both the search term and the field to lowercase using LOWER() function.

            When dealing with grades, use a WITH clause to create grade_values that converts climbing grades to numeric values for proper sorting:
            5.10a = 100, 5.10b = 101, 5.10c = 102, 5.10d = 103
            5.11a = 110, 5.11b = 111, 5.11c = 112, 5.11d = 113
            5.12a = 120, 5.12b = 121, 5.12c = 122, 5.12d = 123
            5.13a = 130, 5.13b = 131, 5.13c = 132, 5.13d = 133
            5.14a = 140, 5.14b = 141, 5.14c = 142, 5.14d = 143

            List of areas in Guadalcazar:
            - San Cayetano
            - Joya del Salitre
            - Zelda
            - Gruta de las Candelas
            - Panales

            Common route types include:
            - sport
            - trad
            - boulder
            - mixed

            Quality ratings are typically:
            - From 10 to 100
            - Eg. "Quality: 90"

            Let's lsit some examples of user prompts:

            EXAMPLE 1:
            User: "Give me all routes from all areas where routes grades are between 11d and 12c"
            Query Result:
            WITH grade_values AS (
            SELECT
              "Route".id,
              "Route".grade,
              CASE
                -- 5.11 grades
                WHEN "Route".grade = '5.11a' THEN 110
                WHEN "Route".grade = '5.11b' THEN 111
                WHEN "Route".grade = '5.11c' THEN 112
                WHEN "Route".grade = '5.11d' THEN 113
                -- 5.12 grades
                WHEN "Route".grade = '5.12a' THEN 120
                WHEN "Route".grade = '5.12b' THEN 121
                WHEN "Route".grade = '5.12c' THEN 122
                -- Handle common alternative notations
                WHEN "Route".grade = '11a' THEN 110
                WHEN "Route".grade = '11b' THEN 111
                WHEN "Route".grade = '11c' THEN 112
                WHEN "Route".grade = '11d' THEN 113
                WHEN "Route".grade = '12a' THEN 120
                WHEN "Route".grade = '12b' THEN 121
                WHEN "Route".grade = '12c' THEN 122
              END as grade_value
            FROM "Route"
          )
          SELECT
            r.*,
            a.name as area_name,
            a.description as area_description
          FROM "Route" r
          JOIN "Area" a ON r."areaId" = a.id
          JOIN grade_values gv ON r.id = gv.id
          WHERE gv.grade_value BETWEEN 113 AND 122  -- From 5.11d to 5.12c
          ORDER BY gv.grade_value, r.name;

          EXAMPLE 2:
          User: "Give me all routes with grade 12.b from "Joya del Salitre" area"
          Query Result:
          WITH grade_values AS (
            SELECT
              "Route".id,
              "Route".grade,
              CASE
                WHEN "Route".grade = '5.12b' THEN 121
                WHEN "Route".grade = '12b' THEN 121
              END as grade_value
            FROM "Route"
          )
          SELECT
            r.*,
            a.name as area_name
          FROM "Route" r
          JOIN "Area" a ON r."areaId" = a.id
          JOIN grade_values gv ON r.id = gv.id
          WHERE a.name = 'Joya del Salitre'
            AND gv.grade_value = 121  -- 5.12b/12b
          ORDER BY r.name;

          EXAMPLE 3:
          User: "Give me all routes with best quality from "San Cayetano" area"
          Query Result:
          WITH grade_values(grade, value) AS (
            VALUES
              ('5.10a', 100), ('5.10b', 101), ('5.10c', 102), ('5.10d', 103),
              ('5.11a', 110), ('5.11b', 111), ('5.11c', 112), ('5.11d', 113),
              ('5.12a', 120), ('5.12b', 121), ('5.12c', 122), ('5.12d', 123),
              ('5.13a', 130), ('5.13b', 131), ('5.13c', 132), ('5.13d', 133),
              ('5.14a', 140), ('5.14b', 141), ('5.14c', 142), ('5.14d', 143)
          ),
          route_quality AS (
            SELECT
              r.*,
              CAST(SUBSTRING(r."quality" FROM 'Quality: (\d+)') AS INTEGER) as quality_value,
              COALESCE(gv.value, 0) as grade_value
            FROM "Route" r
            LEFT JOIN "Area" a ON r."areaId" = a."id"
            LEFT JOIN grade_values gv ON r."grade" = gv.grade
            WHERE a."name" = 'San Cayetano'
          ),
          max_quality AS (
            SELECT MAX(quality_value) as max_quality_value
            FROM route_quality
          )
          SELECT
            rq."name",
            rq.description,
            rq."grade",
            rq."quality",
            rq."type",
            rq."length",
            rq."bolts"
          FROM route_quality rq, max_quality mq
          WHERE rq.quality_value = mq.max_quality_value
          ORDER BY rq.grade_value ASC;

          Remember to handle NULL values appropriately and use proper joins between route and area tables.`,
          prompt: `Generate the query necessary to retrieve the data the user wants: ${input}`,
          schema: z.object({
            query: z.string(),
          }),
        });
        console.log("Query result:", result.object.query);
        return result.object.query;
      } catch (e) {
        console.error(e);
        throw new Error("Failed to generate query");
      }
    }),
  runQuery: publicProcedure
    .input(z.object({ query: z.string() }))
    .mutation(async ({ input, ctx }) => {
      const normalizedQuery = input.query.trim().toLowerCase();

      // Check if query starts with either SELECT or WITH
      // if (
      //   !(
      //     normalizedQuery.startsWith("select") ||
      //     normalizedQuery.startsWith("with")
      //   ) ||
      //   normalizedQuery.includes("drop") ||
      //   normalizedQuery.includes("delete") ||
      //   normalizedQuery.includes("insert") ||
      //   normalizedQuery.includes("update") ||
      //   normalizedQuery.includes("alter") ||
      //   normalizedQuery.includes("truncate") ||
      //   normalizedQuery.includes("create") ||
      //   normalizedQuery.includes("grant") ||
      //   normalizedQuery.includes("revoke")
      // ) {
      //   throw new Error(
      //     "Only SELECT queries (including WITH clauses) are allowed",
      //   );
      // }

      // For WITH queries, ensure they eventually contain a SELECT
      if (
        normalizedQuery.startsWith("with") &&
        !normalizedQuery.includes("select")
      ) {
        throw new Error("WITH clauses must be followed by a SELECT statement");
      }

      let data: any;
      try {
        data = await ctx.db.$queryRawUnsafe(input.query);
      } catch (e: any) {
        debugger;
        console.error("Database error:", e);
        if (e.message.includes('relation "unicorns" does not exist')) {
          console.log(
            "Table does not exist, creating and seeding it with dummy data now...",
          );
          throw Error("Table does not exist");
        } else {
          throw e;
        }
      }
      debugger;
      return data;
    }),
});
